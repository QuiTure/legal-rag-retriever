import json
import re
import logging
import requests
import numpy as np
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_openai import OpenAIEmbeddings

# 假设 config 模块已正确配置
import config

logger = logging.getLogger(__name__)


class LegalProvisionSearcher:
    """
    法律条文检索器。
    采用两阶段检索架构：
    1. 初排 (Recall): 基于本地 Numpy 矩阵计算向量余弦相似度，进行快速召回。
    2. 精排 (Rerank): 多线程调用外部 Reranker API 对初排结果进行打分重排。
    """

    def __init__(
            self,
            json_path: str,
            api_key: str = config.EMBEDDING_API_KEY,
            base_url: str = config.EMBEDDING_BASE_URL,
            model_name: str = config.EMBEDDING_MODEL_NAME,
            dimensions: int = 1024,
            reranker_api_key: str = config.RERANKER_API_KEY,
            reranker_base_url: str = config.RERANKER_BASE_URL,
            reranker_model_name: str = config.RERANKER_MODEL_NAME
    ) -> None:
        self.json_path = json_path
        self.dimensions = dimensions

        # Reranker 参数
        self.reranker_api_key = reranker_api_key
        self.reranker_base_url = reranker_base_url
        self.reranker_model_name = reranker_model_name

        # 初始化 HTTP Session 以复用底层 TCP 连接，提升并发请求性能
        self.http_session = requests.Session()
        self.http_session.headers.update({
            "Authorization": f"Bearer {self.reranker_api_key}",
            "Content-Type": "application/json"
        })

        # 初始化 Embedding 客户端
        self.embeddings = OpenAIEmbeddings(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            dimensions=self.dimensions
        )

        # 加载数据并构建向量矩阵
        self.data, self.doc_vectors, self.doc_norms = self._load_and_build_index()

    def _load_and_build_index(self) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """加载 JSON 数据，并将向量转换为 Numpy 矩阵供快速检索"""
        path = Path(self.json_path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件未找到: {self.json_path}")

        with open(path, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {e}")

        if not isinstance(raw_data, list):
            raise ValueError("数据格式错误：期望为扁平化的列表 (List[dict])。")

        valid_data = []
        vectors = []

        for item in raw_data:
            vec = item.get("向量")
            if vec and isinstance(vec, list) and len(vec) == self.dimensions:
                valid_data.append(item)
                vectors.append(vec)

        if not valid_data:
            logger.warning(f"文件 {self.json_path} 中未找到符合维度要求的向量数据。")

        # 转换为 Float32 矩阵以加速计算
        doc_vectors = np.array(vectors, dtype=np.float32)

        # 预计算所有法条向量的模长 (Norm)
        if doc_vectors.size > 0:
            doc_norms = np.linalg.norm(doc_vectors, axis=1)
            doc_norms[doc_norms == 0] = 1e-9  # 防止除以 0
        else:
            doc_norms = np.array([])

        logger.info(f"成功加载 {len(valid_data)} 条法条向量数据。")
        return valid_data, doc_vectors, doc_norms

    def _format_provision(self, item: Dict[str, Any]) -> str:
        """格式化法条名称"""
        keys = ["法律", "编", "编名", "章", "章名", "节", "节名", "条", "款", "项"]
        parts = [str(item.get(k, "")) for k in keys if item.get(k)]
        return " ".join(parts).strip()

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """调用 Reranker 接口对候选文档进行重排"""
        if not candidates:
            return []

        url = f"{self.reranker_base_url.rstrip('/')}/rerank"
        documents = [c.get("content", "") for c in candidates]

        payload = {
            "model": self.reranker_model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k
        }

        try:
            response = self.http_session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            res_data = response.json()

            results = res_data.get("results", [])
            reranked_candidates = []

            for r in results:
                idx = r.get("index", 0)
                if idx < len(candidates):
                    candidate = candidates[idx].copy()
                    candidate["score"] = r.get("relevance_score", 0.0)
                    reranked_candidates.append(candidate)

            reranked_candidates.sort(key=lambda x: x["score"], reverse=True)
            return reranked_candidates

        except requests.RequestException as e:
            logger.error(f"Rerank 步骤请求失败: {e}。将回退使用初排向量分数。")
            return candidates[:top_k]

    def search(self, query: str, top_k: int = 5, retrieve_k: int = 50) -> List[Dict[str, Any]]:
        """
        执行多关键词混合搜索。

        Args:
            query (str): 搜索语句（支持逗号分隔多个关键词）。
            top_k (int): 最终返回的条目数量。
            retrieve_k (int): 每个关键词在初排阶段召回的数量。

        Returns:
            List[Dict]: 检索并重排后的法条结果列表。
        """
        if self.doc_vectors.size == 0:
            return []

        # 1. 切分字符串获取关键词列表
        keywords = [k.strip() for k in re.split(r'[,，]', query) if k.strip()]
        if not keywords:
            return []

        # ================= 第一阶段：批量并行 Embedding & 矩阵运算 (召回) =================
        try:
            query_vecs = self.embeddings.embed_documents(keywords)
            query_matrix = np.array(query_vecs, dtype=np.float32)
        except Exception as e:
            logger.error(f"向量化接口调用失败: {e}", exc_info=True)
            return []

        query_norms = np.linalg.norm(query_matrix, axis=1)
        query_norms[query_norms == 0] = 1e-9

        # 矩阵乘法快速计算余弦相似度
        sim_matrix = np.dot(query_matrix, self.doc_vectors.T) / np.outer(query_norms, self.doc_norms)

        actual_retrieve_k = min(retrieve_k, sim_matrix.shape[1])
        top_indices = np.argpartition(-sim_matrix, actual_retrieve_k - 1, axis=1)[:, :actual_retrieve_k]

        all_candidates = []
        for i, kw in enumerate(keywords):
            kw_indices = top_indices[i]
            sorted_kw_indices = kw_indices[np.argsort(-sim_matrix[i, kw_indices])]

            candidates = [{
                "score": float(sim_matrix[i, idx]),
                "content": self.data[idx].get("内容", ""),
                "raw_item": self.data[idx]
            } for idx in sorted_kw_indices if float(sim_matrix[i, idx]) > 0]

            all_candidates.append((kw, candidates))

        # ================= 第二阶段：多线程并发 Rerank (精排) =================
        merged_results: Dict[str, Dict[str, Any]] = {}

        def process_rerank(kw_and_cands: Tuple[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            kw, cands = kw_and_cands
            return self._rerank(kw, cands, top_k)

        max_workers = min(10, len(keywords))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_kw = {executor.submit(process_rerank, item): item[0] for item in all_candidates}

            for future in concurrent.futures.as_completed(future_to_kw):
                try:
                    final_results = future.result()
                    # 去重与合并逻辑
                    for res in final_results:
                        provision_name = self._format_provision(res["raw_item"])
                        current_score = res["score"]

                        if provision_name not in merged_results or current_score > merged_results[provision_name][
                            "相似度"]:
                            merged_results[provision_name] = {
                                "法条": provision_name,
                                "内容": res["content"],
                                "相似度": current_score
                            }
                except Exception as e:
                    kw = future_to_kw[future]
                    logger.error(f"处理关键词 '{kw}' 的精排任务时发生异常: {e}", exc_info=True)

        # 全局按分数降序排列并截取 Top-K
        output = list(merged_results.values())
        output.sort(key=lambda x: x["相似度"], reverse=True)

        return output[:top_k]