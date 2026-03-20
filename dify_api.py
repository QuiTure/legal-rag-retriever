from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import traceback

# 导入你的查询类
from utils.legal_provision_searcher import LegalProvisionSearcher

app = FastAPI(title="Dify 外部知识库适配-刑法RAG")

# ===================== 配置区 =====================
API_KEY = "my-dify-api-key-496a8s61dg6as5d4g96as1sgdqsahfgsa168asg6"
KNOWLEDGE_ID = "刑法RAG数据库_远程"
# ===================================================

# 初始化数据库
db_query = LegalProvisionSearcher("database/legal_provisions/中华人民共和国刑法.json")

# Dify 请求模型（完整兼容官方）
class RetrievalSetting(BaseModel):
    top_k: int
    score_threshold: float

class MetadataCondition(BaseModel):
    logical_operator: Optional[str] = "and"
    conditions: Optional[List[Dict]] = None

class RetrievalRequest(BaseModel):
    knowledge_id: str
    query: str
    retrieval_setting: RetrievalSetting
    metadata_condition: Optional[MetadataCondition] = None

# 工具函数：法条去重（避免重复返回）
def deduplicate_records(records: List[Dict]) -> List[Dict]:
    seen = set()
    new_records = []
    for item in records:
        # 用法条内容+标题唯一标识去重
        key = (item["title"], item["content"])
        if key not in seen:
            seen.add(key)
            new_records.append(item)
    return new_records

# 核心接口
@app.post("/retrieval")
async def retrieval(
    request: RetrievalRequest,
    authorization: str = Header(None)
):
    try:
        # ===================== 修复1：安全鉴权 =====================
        if not authorization:
            raise HTTPException(
                status_code=403,
                detail={"error_code": 1001, "error_msg": "缺少Authorization请求头"}
            )
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=403,
                detail={"error_code": 1001, "error_msg": "Authorization格式错误，必须以Bearer 开头"}
            )
        token = authorization.split(" ", 1)[1]
        if token != API_KEY:
            raise HTTPException(
                status_code=403,
                detail={"error_code": 1002, "error_msg": "API-Key授权失败"}
            )

        # 校验知识库ID
        if request.knowledge_id != KNOWLEDGE_ID:
            raise HTTPException(
                status_code=404,
                detail={"error_code": 2001, "error_msg": "知识库不存在"}
            )

        # 空查询处理
        if not request.query.strip():
            return {"records": []}

        # ===================== 多关键词查询 =====================
        query_list = [q.strip() for q in request.query.split(",") if q.strip()]
        raw_results = []
        for q in query_list:
            raw_results.extend(db_query.search(q, request.retrieval_setting.top_k))

        # ===================== 格式转换 =====================
        dify_records = []
        for item in raw_results:
            score = float(item.get("score", 0.9))  # 确保为数字
            dify_records.append({
                "content": item.get("内容", ""),
                "score": round(score, 2),  # 保留两位小数
                "title": item.get("法条", "未知法条"),
                "metadata": {"source": "中华人民共和国刑法数据库"}
            })

        # ===================== 修复2：去重 =====================
        dify_records = deduplicate_records(dify_records)

        # ===================== 修复3：按分数降序排序（核心！） =====================
        dify_records.sort(key=lambda x: x["score"], reverse=True)

        # ===================== 分数过滤 + TOPK截断 =====================
        filtered_records = [
            r for r in dify_records
            if r["score"] >= request.retrieval_setting.score_threshold
        ]
        final_records = filtered_records[:request.retrieval_setting.top_k]

        return {"records": final_records}

    # ===================== 修复4：全局异常捕获 =====================
    except Exception as e:
        print(f"服务异常：{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={"error_code": 5000, "error_msg": f"服务器内部错误：{str(e)}"}
        )