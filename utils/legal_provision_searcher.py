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


class LegalProvisionSearcher:
    """
    法律条文检索器（适配“向量组”结构）。
    三阶段架构：
    0. 路由 (Routing): LLM 从本地法律文件中筛选候选文件。
    1. 初排 (Recall): 对每条法条的“向量组”计算相似度，取 max 作为该法条初排分。
       - 向量组格式: [整句向量, 子句向量1, 子句向量2, ...]
    2. 精排 (Rerank): 仍然基于整句文本 content 调用外部 Reranker。
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
        self.db_dir = db_dir or (Path(__file__).resolve().parent.parent / "database" / "legal_provisions")
        self.max_cache_size = max_cache_size

        # 缓存结构:
        # {filename: (valid_data, group_max_vectors, group_max_norms, group_ranges)}
        # - valid_data: 每条法条原始 item（与 group_ranges 一一对应）
        # - group_max_vectors / group_max_norms: 将所有法条的“向量组”展平成一个大矩阵
        # - group_ranges: 每条法条在大矩阵中的切片范围 (start, end)
        self.loaded_laws_cache: OrderedDict[
            str,
            Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[Tuple[int, int]]]
        ] = OrderedDict()

        self.reranker_api_key = reranker_api_key or config.RERANKER_API_KEY
        self.reranker_base_url = reranker_base_url or config.RERANKER_BASE_URL
        self.reranker_model_name = reranker_model_name or config.RERANKER_MODEL_NAME

        self.http_session = requests.Session()
        self.http_session.headers.update({
            "Authorization": f"Bearer {self.reranker_api_key}",
            "Content-Type": "application/json"
        })

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

        self.embeddings = OpenAIEmbeddings(
            base_url=embedding_base_url or config.EMBEDDING_BASE_URL,
            api_key=embedding_api_key or config.EMBEDDING_API_KEY,
            model=embedding_model_name or config.EMBEDDING_MODEL_NAME,
            dimensions=self.embedding_dimensions
        )

    def _is_single_vector(self, vec: Any) -> bool:
        """判断是否为单个合法向量。"""
        return isinstance(vec, list) and len(vec) == self.embedding_dimensions and all(
            isinstance(x, (int, float)) for x in vec
        )

    def _normalize_vector_group(self, vec_field: Any) -> List[List[float]]:
        """
        将 item['向量'] 统一归一为“向量组”格式:
        - 旧格式: 单向量 [d] -> [[d]]
        - 新格式: 向量组 [[d], [d], ...]
        """
        if self._is_single_vector(vec_field):
            return [vec_field]

        if isinstance(vec_field, list):
            group = []
            for v in vec_field:
                if self._is_single_vector(v):
                    group.append(v)
            return group

        return []

    def _load_and_build_index(
            self,
            json_path: Path
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        加载指定 JSON，并建立适配“向量组”的索引：
        - valid_data: 有效法条条目
        - flat_vectors: 所有法条向量组展平后的矩阵 (T, D)
        - flat_norms: 展平矩阵每行范数 (T,)
        - group_ranges: 每条法条在 flat_vectors 中对应的 [start, end)
        """
        if not json_path.exists():
            raise FileNotFoundError(f"数据文件未找到: {json_path}")

        start_time = time.time()
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {e}")

        if not isinstance(raw_data, list):
            raise ValueError("数据格式错误：期望为扁平化的列表 (List[dict])。")

        valid_data: List[Dict[str, Any]] = []
        flat_vectors_list: List[List[float]] = []
        group_ranges: List[Tuple[int, int]] = []

        cursor = 0
        for item in raw_data:
            vec_group = self._normalize_vector_group(item.get("向量"))

            if not vec_group:
                continue

            start = cursor
            flat_vectors_list.extend(vec_group)
            cursor += len(vec_group)
            end = cursor

            valid_data.append(item)
            group_ranges.append((start, end))

        if not valid_data:
            logger.warning(f"文件 {json_path} 中未找到符合维度要求的向量数据（含向量组）。")

        flat_vectors = np.array(flat_vectors_list, dtype=np.float32)
        if flat_vectors.size > 0:
            flat_norms = np.linalg.norm(flat_vectors, axis=1)
            flat_norms[flat_norms == 0] = 1e-9
        else:
            flat_norms = np.array([])

        logger.info(
            f"成功加载 {json_path.name}: {len(valid_data)} 条法条, "
            f"{len(flat_vectors_list)} 个子向量, 耗时 {time.time() - start_time:.2f}s。"
        )
        return valid_data, flat_vectors, flat_norms, group_ranges

    def _get_cached_law(
            self,
            law_file: str
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """带 LRU 剔除机制的缓存读取。"""
        if law_file in self.loaded_laws_cache:
            self.loaded_laws_cache.move_to_end(law_file)
            return self.loaded_laws_cache[law_file]

        law_path = self.db_dir / law_file
        data = self._load_and_build_index(law_path)

        self.loaded_laws_cache[law_file] = data
        self.loaded_laws_cache.move_to_end(law_file)

        if len(self.loaded_laws_cache) > self.max_cache_size:
            evicted = self.loaded_laws_cache.popitem(last=False)
            logger.info(f"达到缓存上限，已释放最早加载的法律文件: {evicted[0]}")

        return data

    def _get_relevant_laws_from_llm(self, query: str) -> List[str]:
        """调用大模型，根据查询内容动态选择可能相关的法条文件。"""
        if not self.db_dir.exists():
            logger.warning(f"数据库目录不存在: {self.db_dir}")
            return []

        available_files = [f.name for f in self.db_dir.glob("*.json")]
        if not available_files:
            return []

        system_prompt = (
            "你是一个法律助理。我将提供一组可用的法律文件名以及用户的查询请求。"
            "请根据用户的查询，从列表中选出所有可��适用的法律文件名。\n"
            "请必须返回 JSON 格式数据，包含一个键 'laws'，其值为你挑选的文件名列表（精确匹配我提供的文件名）。"
        )

        user_prompt = f"可用的法律文件：\n{json.dumps(available_files, ensure_ascii=False)}\n\n用户的查询：\n{query}"

        start_time = time.time()
        try:
            response = self.model.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            cost_time = time.time() - start_time

            result_json = json.loads(response.content)
            selected_laws = result_json.get("laws", [])
            valid_selected = [law for law in selected_laws if law in available_files]

            logger.info(f"[LLM路由] 耗时 {cost_time:.2f}s, 针对查询 '{query}' 筛选出的法律文件: {valid_selected}")
            return valid_selected
        except Exception as e:
            logger.error(f"[LLM路由] 筛选法律文件失败: {e}", exc_info=True)
            return []

    def _format_provision(self, item: Dict[str, Any]) -> str:
        """格式化法条名称。"""
        keys = ["法律", "编", "编名", "章", "章名", "节", "节名", "条", "款", "项"]
        parts = [str(item.get(k, "")) for k in keys if item.get(k)]
        return " ".join(parts).strip()

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        调用 Reranker 接口对候选文档进行重排。
        注意：这里仍使用 candidates 的整句 content（满足你的要求）。
        """
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

    def search(self, query: str, top_k: int = 10, retrieve_k: int = 50, score: float = 0.0) -> List[Dict[str, Any]]:
        """执行多关键词混合搜索。"""
        total_start = time.time()
        logger.info(f"开始执行检索管道，Query: '{query}'")

        # 1) LLM 路由
        selected_law_files = self._get_relevant_laws_from_llm(query)
        if not selected_law_files:
            logger.warning("没有匹配到相关的法律文件，检索结束。")
            return []

        combined_data: List[Dict[str, Any]] = []
        combined_flat_vectors_list: List[np.ndarray] = []
        combined_flat_norms_list: List[np.ndarray] = []
        combined_group_ranges: List[Tuple[int, int]] = []  # 全局范围（对应 combined_data）

        # 用于把每个文件内的局部索引，映射到全局展平矩阵索引
        global_flat_cursor = 0

        for law_file in selected_law_files:
            v_data, v_flat_vecs, v_flat_norms, v_group_ranges = self._get_cached_law(law_file)

            if not v_data:
                continue

            combined_data.extend(v_data)

            if v_flat_vecs.size > 0:
                combined_flat_vectors_list.append(v_flat_vecs)
                combined_flat_norms_list.append(v_flat_norms)

                # 将文件内 group_ranges 偏移到全局
                for s, e in v_group_ranges:
                    combined_group_ranges.append((s + global_flat_cursor, e + global_flat_cursor))

                global_flat_cursor += v_flat_vecs.shape[0]
            else:
                # 正常情况下不会出现（v_data 有值却无向量），做保护
                combined_group_ranges.extend([(0, 0)] * len(v_data))

        if not combined_flat_vectors_list or not combined_group_ranges:
            return []

        flat_doc_vectors = np.vstack(combined_flat_vectors_list)   # (T, D)
        flat_doc_norms = np.concatenate(combined_flat_norms_list)  # (T,)

        keywords = [k.strip() for k in re.split(r'[,，]', query) if k.strip()]
        if not keywords:
            return []

        # 2) 查询向量化
        embed_start = time.time()
        try:
            query_vecs = self.embeddings.embed_documents(keywords)
            query_matrix = np.array(query_vecs, dtype=np.float32)  # (K, D)
            logger.info(f"[向量化] 耗时 {time.time() - embed_start:.2f}s, 成功生成 {len(keywords)} 个关键词向量。")
        except Exception as e:
            logger.error(f"[向量化] 接口调用失败: {e}", exc_info=True)
            return []

        query_norms = np.linalg.norm(query_matrix, axis=1)
        query_norms[query_norms == 0] = 1e-9

        # 3) 初排：先算关键词 vs 全部子向量，再按每条��条取 max
        # sim_sub: (K, T)
        sim_sub = np.dot(query_matrix, flat_doc_vectors.T) / np.outer(query_norms, flat_doc_norms)

        # sim_doc: (K, N_docs)  每个元素是“该关键词对该法条向量组的最大相似度”
        n_docs = len(combined_data)
        sim_doc = np.full((len(keywords), n_docs), -1.0, dtype=np.float32)

        for doc_idx, (s, e) in enumerate(combined_group_ranges):
            if s < e:
                sim_doc[:, doc_idx] = np.max(sim_sub[:, s:e], axis=1)

        actual_retrieve_k = min(retrieve_k, sim_doc.shape[1])
        if actual_retrieve_k <= 0:
            return []

        top_indices = np.argpartition(-sim_doc, actual_retrieve_k - 1, axis=1)[:, :actual_retrieve_k]

        all_candidates = []
        for i, kw in enumerate(keywords):
            kw_indices = top_indices[i]
            sorted_kw_indices = kw_indices[np.argsort(-sim_doc[i, kw_indices])]

            candidates = [{
                "score": float(sim_doc[i, idx]),  # 初排分 = 该法条向量组 max 相似度
                "content": combined_data[idx].get("内容", ""),  # rerank 用整句内容
                "raw_item": combined_data[idx]
            } for idx in sorted_kw_indices if float(sim_doc[i, idx]) > 0]

            all_candidates.append((kw, candidates))

        merged_results: Dict[str, Dict[str, Any]] = {}

        def process_rerank(kw_and_cands: Tuple[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            kw, cands = kw_and_cands
            return self._rerank(kw, cands, top_k)

        # 4) 精排（并发）
        rerank_start = time.time()
        max_workers = min(10, len(keywords))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_kw = {executor.submit(process_rerank, item): item[0] for item in all_candidates}

            for future in concurrent.futures.as_completed(future_to_kw):
                try:
                    final_results = future.result()
                    for res in final_results:
                        provision_name = self._format_provision(res["raw_item"])
                        current_score = res["score"]

                        if current_score < score:
                            continue

                        if provision_name not in merged_results or current_score > merged_results[provision_name]["相似度"]:
                            merged_results[provision_name] = {
                                "法条": provision_name,
                                "内容": res["content"],
                                "相似度": current_score
                            }
                except Exception as e:
                    kw = future_to_kw[future]
                    logger.error(f"处理关键词 '{kw}' 的精排任务时发生异常: {e}", exc_info=True)

        logger.info(f"[精排阶段] 多线程 Rerank 耗时 {time.time() - rerank_start:.2f}s。")

        output = list(merged_results.values())
        output.sort(key=lambda x: x["相似度"], reverse=True)
        final_output = output[:top_k]

        logger.info(f"检索管道执行完毕，总耗时 {time.time() - total_start:.2f}s，最终返回 {len(final_output)} 条结果。")
        return final_output