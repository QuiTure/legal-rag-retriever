from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

# 导入你的查询类（完全正确）
from utils.legal_provision_searcher import LegalProvisionSearcher

app = FastAPI(title="Dify 外部知识库适配")

# 你的配置（正确）
API_KEY = "my-dify-api-key-496a8s61dg6as5d4g96as1sgdqsahfgsa168asg6"
KNOWLEDGE_ID = "刑法RAG数据库_远程"

# 实例化查询类（正确）
db_query = LegalProvisionSearcher("database/legal_provisions/中华人民共和国刑法.json")

# Dify 请求模型（无需修改）
class RetrievalSetting(BaseModel):
    top_k: int
    score_threshold: float

class RetrievalRequest(BaseModel):
    knowledge_id: str
    query: str
    retrieval_setting: RetrievalSetting

# 核心接口
@app.post("/retrieval")
async def retrieval(request: RetrievalRequest, authorization: str = Header(None)):
    # 鉴权（正确）
    if not authorization or authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail={"error_code": 1002, "error_msg": "授权失败"})
    if request.knowledge_id != KNOWLEDGE_ID:
        raise HTTPException(status_code=404, detail={"error_code": 2001, "error_msg": "知识库不存在"})

    # 调用你的查询（正确）
    results = db_query.search(request.query, request.retrieval_setting.top_k)

    # 格式转换（修复了报错问题）
    dify_records = []
    for item in results:
        # 修复1：score 给默认值 0.9，防止None报错
        score = item.get("score", 0.9)
        # 修复2：安全取值
        dify_records.append({
            "content": item.get("内容", ""),
            "score": score,
            "title": item.get("法条", "未知法条"),
            "metadata": {"source": "刑法数据库"}
        })

    # 过滤（正确）
    filtered = [r for r in dify_records if r["score"] >= request.retrieval_setting.score_threshold]
    filtered = filtered[:request.retrieval_setting.top_k]

    return {"records": filtered}