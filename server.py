from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.legal_provision_searcher import LegalProvisionSearcher

app = FastAPI(title="法律搜索API")

# 初始化搜索器 (全局加载一次即可)
searcher = LegalProvisionSearcher(json_path="database/legal_provisions/中华人民共和国刑法.json")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_provisions(req: SearchRequest):
    try:
        results = searcher.search(query=req.query, top_k=req.top_k)
        return {"code": 200, "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))