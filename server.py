import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

from utils.legal_provision_searcher import LegalProvisionSearcher
from utils.legal_case_searcher import LegalCaseSearcher

# ================= 1. 日志全局配置 =================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_format = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

file_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "app.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(log_format)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)

for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    u_logger = logging.getLogger(logger_name)
    u_logger.handlers = []
    u_logger.propagate = True

logger = logging.getLogger(__name__)
# =================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在启动服务：初始化搜索器...")

    try:
        # 法律条文搜索器
        app.state.provision_searcher = LegalProvisionSearcher()
        logger.info("法律条文搜索器初始化完成。")

        # 法律案例搜索器
        app.state.case_searcher = LegalCaseSearcher()
        logger.info("法律案例搜索器初始化完成。")

    except Exception as e:
        logger.error(f"搜索器初始化失败: {e}", exc_info=True)
        raise e

    yield

    logger.info("正在关闭服务：清理连接池...")
    try:
        if hasattr(app.state, "provision_searcher") and hasattr(app.state.provision_searcher, "http_session"):
            app.state.provision_searcher.http_session.close()

        if hasattr(app.state, "case_searcher") and hasattr(app.state.case_searcher, "http_session"):
            app.state.case_searcher.http_session.close()
    finally:
        logger.info("资源清理完毕。")


app = FastAPI(title="法律数据库搜索API", lifespan=lifespan)


class SearchRequest(BaseModel):
    db_type: Literal["legal_provisions", "legal_cases"]
    query: str
    top_k: int = 10
    score: float = 0.5


@app.post("/search")
def search(req: SearchRequest, request: Request):
    logger.info(f"收到检索请求: db_type={req.db_type}, query='{req.query}', top_k={req.top_k}, score={req.score}")
    try:
        if req.db_type == "legal_provisions":
            searcher: LegalProvisionSearcher = request.app.state.provision_searcher
            results = searcher.search(query=req.query, top_k=req.top_k, score=req.score)

        elif req.db_type == "legal_cases":
            searcher: LegalCaseSearcher = request.app.state.case_searcher
            results = searcher.search(query=req.query, top_k=req.top_k, score=req.score)

        else:
            raise ValueError(f"暂不支持的数据库类型: {req.db_type}")

        logger.info(f"检索成功，共返回 {len(results)} 条结果。")
        return {
            "code": 200,
            "msg": "success",
            "data": results
        }

    except Exception as e:
        logger.error(f"检索接口发生异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))