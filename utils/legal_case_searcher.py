import json
import re
import logging
import requests
import numpy as np
import concurrent.futures
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import OrderedDict

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import config

logger = logging.getLogger(__name__)


class LegalCaseSearcher:
    """
    法律案例检索器（与 LegalProvisionSearcher 同逻辑样式）：
    0. 路由(Routing): LLM 从本地案例文件中选择相关文件
    1. 初排(Recall): 基于 Numpy 余弦相似度召回
    2. 精排(Rerank): 多线程调用 Reranker 接口重排
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
        reranker_api_key: Optional[str] = None,
        reranker_base_url: Optional[str] = None,
        reranker_model_name: Optional[str] = None,
        db_dir: Optional[Path] = None,
        max_cache_size: int = 5
    ) -> None:
        self.embedding_dimensions = embedding_dimensions or config.EMBEDDING_DIMENSIONS
        self.db_dir = db_dir or (Path(__file__).resolve().parent.parent / "database" / "legal_case")
        self.max_cache_size = max_cache_size

        # LRU 缓存：{filename: (valid_data, doc_vectors, doc_norms)}
        self.loaded_case_cache: OrderedDict[str, Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]] = OrderedDict()

        # Reranker
        self.reranker_api_key = reranker_api_key or config.RERANKER_API_KEY
        self.reranker_base_url = reranker_base_url or config.RERANKER_BASE_URL
        self.reranker_model_name = reranker_model_name or config.RERANKER_MODEL_NAME

        self.http_session = requests.Session()
        self.http_session.headers.update({
            "Authorization": f"Bearer {self.reranker_api_key}",
            "Content-Type": "application/json"
        })

        # LLM 路由
        self.model = ChatOpenAI(
            base_url=llm_base_url or config.LLM_BASE_URL,
            api_key=llm_api_key or config.LLM_API_KEY,
            model=llm_model_name or config.LLM_MODEL_NAME,
            temperature=0,
            top_p=0.7,
            seed=42,
            model_kwargs={
                "response_format": {"type": "json_object"}
            }
        )

        # Embedding
        self.embeddings = OpenAIEmbeddings(
            base_url=embedding_base_url or config.EMBEDDING_BASE_URL,
            api_key=embedding_api_key or config.EMBEDDING_API_KEY,
            model=embedding_model_name or config.EMBEDDING_MODEL_NAME,
            dimensions=self.embedding_dimensions
        )

    # -------------------------- 数据加载与缓存 --------------------------
    def _load_and_build_index(self, json_path: Path) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """加载指定案例文件并构建向量索引"""
        if not json_path.exists():
            raise FileNotFoundError(f"数据文件未找到: {json_path}")

        start_time = time.time()
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {e}")

        if not isinstance(raw_data, list):
            raise ValueError(f"数据格式错误：{json_path.name} 期望为 List[dict]")

        valid_data = []
        vectors = []

        for item in raw_data:
            vec = item.get("违法行为向量")
            if vec and isinstance(vec, list) and len(vec) == self.embedding_dimensions:
                valid_data.append(item)
                vectors.append(vec)

        if not valid_data:
            logger.warning(f"文件 {json_path.name} 中未找到符合维度要求的 '违法行为向量'。")

        doc_vectors = np.array(vectors, dtype=np.float32)

        if doc_vectors.size > 0:
            doc_norms = np.linalg.norm(doc_vectors, axis=1)
            doc_norms[doc_norms == 0] = 1e-9
        else:
            doc_norms = np.array([])

        logger.info(f"成功加载 {json_path.name}: {len(valid_data)} 条案例, 耗时 {time.time() - start_time:.2f}s")
        return valid_data, doc_vectors, doc_norms

    def _get_cached_case_file(self, case_file: str) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """带 LRU 剔除的缓存读取"""
        if case_file in self.loaded_case_cache:
            self.loaded_case_cache.move_to_end(case_file)
            return self.loaded_case_cache[case_file]

        case_path = self.db_dir / case_file
        data = self._load_and_build_index(case_path)

        self.loaded_case_cache[case_file] = data
        self.loaded_case_cache.move_to_end(case_file)

        if len(self.loaded_case_cache) > self.max_cache_size:
            evicted = self.loaded_case_cache.popitem(last=False)
            logger.info(f"达到缓存上限，已释放最早加载的案例文件: {evicted[0]}")

        return data

    # -------------------------- 路由 --------------------------
    def _get_relevant_case_files_from_llm(self, query: str) -> List[str]:
        """LLM 动态选择可能相关的案例文件"""
        if not self.db_dir.exists():
            logger.warning(f"案例库目录不存在: {self.db_dir}")
            return []

        available_files = [f.name for f in self.db_dir.glob("*.json")]
        if not available_files:
            return []

        system_prompt = (
            "你是一个法律案例检索路由助手。"
            "我将提供可用的案例文件名和用户查询。"
            "请选出所有可能相关的案例文件名。\n"
            "你必须只返回 JSON 对象，格式为："
            "{\"files\": [\"文件名1.json\", \"文件名2.json\"]}。"
            "文件名必须与提供列表完全一致。"
        )
        user_prompt = f"可用案例文件：\n{json.dumps(available_files, ensure_ascii=False)}\n\n用户查询：\n{query}"

        start_time = time.time()
        try:
            response = self.model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            cost = time.time() - start_time

            result_json = json.loads(response.content)
            selected = result_json.get("files", [])
            valid_selected = [x for x in selected if x in available_files]

            logger.info(f"[LLM路由] 耗时 {cost:.2f}s, 查询 '{query}' 命中文件: {valid_selected}")
            return valid_selected
        except Exception as e:
            logger.error(f"[LLM路由] 筛选案例文件失败: {e}", exc_info=True)
            return []

    # -------------------------- 精排辅助 --------------------------
    def _build_rerank_doc(self, item: Dict[str, Any]) -> str:
        """构造用于 reranker 的文档文本"""
        keywords = item.get("关键词", [])
        if isinstance(keywords, list):
            keywords_text = "，".join([str(k) for k in keywords])
        else:
            keywords_text = str(keywords)

        return (
            f"标题：{item.get('标题', '')}\n"
            f"案号：{item.get('案号', '')}\n"
            f"关键词：{keywords_text}\n"
            f"违法行为：{item.get('违法行为', '')}\n"
            f"裁判原则：{item.get('裁判原则', '')}\n"
            f"基本案情：{item.get('基本案情', '')}\n"
            f"裁判理由：{item.get('裁判理由', '')}\n"
            f"法律适用：{item.get('法律适用', '')}"
        )

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
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
            response = self.http_session.post(url, json=payload, timeout=15)
            response.raise_for_status()
            res_data = response.json()

            results = res_data.get("results", [])
            reranked = []

            for r in results:
                idx = r.get("index", 0)
                if idx < len(candidates):
                    c = candidates[idx].copy()
                    c["score"] = float(r.get("relevance_score", 0.0))
                    reranked.append(c)

            reranked.sort(key=lambda x: x["score"], reverse=True)
            return reranked
        except requests.RequestException as e:
            logger.error(f"Rerank 请求失败: {e}。回退使用初排分数。")
            return candidates[:top_k]

    # -------------------------- 主搜索 --------------------------
    def search(self, query: str, top_k: int = 10, retrieve_k: int = 50, score: float = 0.0) -> List[Dict[str, Any]]:
        total_start = time.time()
        logger.info(f"开始执行案例检索，Query: '{query}'")

        # 1) LLM 路由
        selected_case_files = self._get_relevant_case_files_from_llm(query)
        if not selected_case_files:
            logger.warning("没有匹配到相关案例文件，检索结束。")
            return []

        combined_data: List[Dict[str, Any]] = []
        combined_vectors_list: List[np.ndarray] = []
        combined_norms_list: List[np.ndarray] = []

        for case_file in selected_case_files:
            v_data, v_vecs, v_norms = self._get_cached_case_file(case_file)
            combined_data.extend(v_data)

            if v_vecs.size > 0:
                combined_vectors_list.append(v_vecs)
                combined_norms_list.append(v_norms)

        if not combined_vectors_list:
            logger.warning("已选案例文件中没有有效向量数据。")
            return []

        doc_vectors = np.vstack(combined_vectors_list)
        doc_norms = np.concatenate(combined_norms_list)

        keywords = [k.strip() for k in re.split(r"[,，]", query) if k.strip()]
        if not keywords:
            return []

        # 2) 向量化 + 初排
        embed_start = time.time()
        try:
            query_vecs = self.embeddings.embed_documents(keywords)
            query_matrix = np.array(query_vecs, dtype=np.float32)
            logger.info(f"[向量化] 耗时 {time.time() - embed_start:.2f}s，成功生成 {len(keywords)} 个关键词向量。")
        except Exception as e:
            logger.error(f"[向量化] 接口调用失败: {e}", exc_info=True)
            return []

        query_norms = np.linalg.norm(query_matrix, axis=1)
        query_norms[query_norms == 0] = 1e-9

        sim_matrix = np.dot(query_matrix, doc_vectors.T) / np.outer(query_norms, doc_norms)

        actual_retrieve_k = min(retrieve_k, sim_matrix.shape[1])
        top_indices = np.argpartition(-sim_matrix, actual_retrieve_k - 1, axis=1)[:, :actual_retrieve_k]

        all_candidates: List[Tuple[str, List[Dict[str, Any]]]] = []
        for i, kw in enumerate(keywords):
            kw_indices = top_indices[i]
            sorted_kw_indices = kw_indices[np.argsort(-sim_matrix[i, kw_indices])]

            candidates = [{
                "score": float(sim_matrix[i, idx]),
                "content": self._build_rerank_doc(combined_data[idx]),
                "raw_item": combined_data[idx]
            } for idx in sorted_kw_indices if float(sim_matrix[i, idx]) > 0]

            all_candidates.append((kw, candidates))

        # 3) 多线程精排 + 去重合并
        merged_results: Dict[str, Dict[str, Any]] = {}

        def process_rerank(kw_and_cands: Tuple[str, List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
            kw, cands = kw_and_cands
            return kw, self._rerank(kw, cands, top_k)

        rerank_start = time.time()
        max_workers = min(10, len(keywords))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_kw = {executor.submit(process_rerank, item): item[0] for item in all_candidates}

            for future in concurrent.futures.as_completed(future_to_kw):
                kw = future_to_kw[future]
                try:
                    _, final_results = future.result()
                    for res in final_results:
                        current_score = float(res["score"])
                        if current_score < score:
                            continue

                        item = res["raw_item"]
                        case_key = f"{item.get('标题', '')}|{item.get('案号', '')}"

                        if case_key not in merged_results or current_score > merged_results[case_key]["相似度"]:
                            merged_results[case_key] = {
                                "标题": item.get("标题", ""),
                                "案号": item.get("案号", ""),
                                "关键词": item.get("关键词", []),
                                "裁判原则": item.get("裁判原则", ""),
                                "基本案情": item.get("基本案情", ""),
                                "裁判理由": item.get("裁判理由", ""),
                                "法律适用": item.get("法律适用", ""),
                                "违法行为": item.get("违法行为", ""),
                                "相似度": current_score,
                                "命中关键词": kw
                            }
                except Exception as e:
                    logger.error(f"处理关键词 '{kw}' 的精排任务异常: {e}", exc_info=True)

        logger.info(f"[精排阶段] 多线程 Rerank 耗时 {time.time() - rerank_start:.2f}s。")

        output = list(merged_results.values())
        output.sort(key=lambda x: x["相似度"], reverse=True)
        final_output = output[:top_k]

        logger.info(f"案例检索成功，总耗时 {time.time() - total_start:.2f}s，最终返回 {len(final_output)} 条。")
        return final_output