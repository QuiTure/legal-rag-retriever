import json
import re
import numpy as np
import requests
from pathlib import Path
from langchain_openai import OpenAIEmbeddings

import config


class LegalProvisionSearcher:
    def __init__(
            self,
            json_path: str,
            api_key=config.EMBEDDING_API_KEY,
            base_url: str = config.EMBEDDING_BASE_URL,
            model_name: str = config.EMBEDDING_MODEL_NAME,
            dimensions: int = 1024,
            reranker_api_key: str = config.RERANKER_API_KEY,
            reranker_base_url: str = config.RERANKER_BASE_URL,
            reranker_model_name: str = config.RERANKER_MODEL_NAME
    ):
        """
        初始化法律条文搜索器 (包含 Embedding 和 Rerank)
        """
        self.json_path = json_path
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.dimensions = dimensions

        # Reranker 参数
        self.reranker_api_key = reranker_api_key
        self.reranker_base_url = reranker_base_url
        self.reranker_model_name = reranker_model_name

        # 初始化 Embedding 客户端
        self.embeddings = OpenAIEmbeddings(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_name,
            dimensions=self.dimensions
        )

        self.data = self._load_data()

    def _load_data(self):
        """加载 JSON 数据"""
        if not Path(self.json_path).exists():
            raise FileNotFoundError(f"文件未找到: {self.json_path}")

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("数据格式错误：期望为扁平化的列表 (List[dict])。")

        return data

    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def _format_provision(self, item: dict) -> str:
        """根据结构化数据的字段拼接法条路径"""
        keys = ["法律", "编", "编名", "章", "章名", "节", "节名", "条", "款", "项"]
        # 按照指定的键提取，并过滤掉空值
        parts = [str(item.get(k, "")) for k in keys if item.get(k)]
        return " ".join(parts).strip()

    def _rerank(self, query: str, candidates: list, top_k: int):
        """
        调用 Reranker 接口进行精确重排序
        """
        if not candidates:
            return []

        url = self.reranker_base_url.rstrip("/") + "/rerank"
        headers = {
            "Authorization": f"Bearer {self.reranker_api_key}",
            "Content-Type": "application/json"
        }

        documents = [c["content"] for c in candidates]

        payload = {
            "model": self.reranker_model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            res_data = response.json()

            results = res_data.get("results", [])
            reranked_candidates = []

            for r in results:
                idx = r["index"]
                score = r["relevance_score"]
                candidate = candidates[idx].copy()
                candidate["score"] = score  # 替换为精排分数
                reranked_candidates.append(candidate)

            reranked_candidates.sort(key=lambda x: x["score"], reverse=True)
            return reranked_candidates

        except Exception as e:
            print(f"Rerank 步骤失败: {e}。将退回使用初排向量分数。")
            return candidates[:top_k]

    def search(self, query: str, top_k: int = 5, retrieve_k: int = 50) -> list:
        """
        执行检索，并返回格式化后的结果列表。对每个关键词分别进行查找。

        :param query: 包含多个关键词的搜索字符串，以中/英文逗号分割
        :param top_k: 每个关键词独立返回的结果数量
        :param retrieve_k: 向量初排召回的候选数量
        :return: 包含法条和内容字典的数组 (已合并去重)
        """
        # 1. 切分字符串获取关键词列表
        keywords = [k.strip() for k in re.split(r'[,，]', query) if k.strip()]

        if not keywords:
            return []

        # 用于存储所有关键词召回的最终结果，使用法条名称作为 key 进行去重
        merged_results = {}

        # 2. 对每个关键词分别进行完整流程的检索
        for kw in keywords:
            try:
                query_vec = self.embeddings.embed_query(kw)
            except Exception as e:
                print(f"向量化接口调用失败 (关键词: {kw}): {e}")
                continue

            # 初排 (Embedding)
            results = []
            for item in self.data:
                item_vec = item.get("向量")
                if item_vec and len(item_vec) > 0:
                    sim = self._cosine_similarity(query_vec, item_vec)
                    if sim > 0:
                        results.append({
                            "score": sim,
                            "content": item.get("内容", ""),
                            "raw_item": item
                        })

            results.sort(key=lambda x: x['score'], reverse=True)
            candidates = results[:retrieve_k]

            # 精排 (Rerank)
            final_results = self._rerank(kw, candidates, top_k)

            # 3. 将当前关键词的结果合并到总结果中（含去重逻辑）
            for res in final_results:
                provision_name = self._format_provision(res["raw_item"])

                # 如果该法条已存在，保留分数最高的那次记录
                if provision_name not in merged_results or res["score"] > merged_results[provision_name]["相似度"]:
                    merged_results[provision_name] = {
                        "法条": provision_name,
                        "内容": res["content"],
                        "相似度": res["score"]
                    }

        # 4. 将合并后的字典转为列表，并按最终的相似度全局降序排列
        output = list(merged_results.values())
        output.sort(key=lambda x: x["相似度"], reverse=True)

        return output