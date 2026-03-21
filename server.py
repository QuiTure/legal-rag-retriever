import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

from utils.legal_provision_searcher import LegalProvisionSearcher

# ================= 1. 日志全局配置 =================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 定义日志格式
log_format = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)

# 文件处理器（按大小轮转，最大 10MB，保留 5 个备份）
file_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, "app.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(log_format)

# 配置根日志记录器 (Root Logger)
logging.basicConfig(
    level=logging.INFO,  # 默认最低输出级别
    handlers=[console_handler, file_handler]
)

# 统一 Uvicorn 的日志格式（如果在 Uvicorn 下运行，防止日志格式不一致）
for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    u_logger = logging.getLogger(logger_name)
    u_logger.handlers = []  # 清除 Uvicorn 默认的 handler
    u_logger.propagate = True  # 将日志向上传递给 Root Logger

# 配置当前模块的 logger
logger = logging.getLogger(__name__)
# =================================================


# 使用 lifespan 替代全局初始化，确保跨 worker 的进程安全并管理资源
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在启动服务：初始化法律条文搜索器...")
    try:
        # 将实例化后的 searcher 挂载到 app.state 上，实现全局单例
        app.state.searcher = LegalProvisionSearcher()
        logger.info("搜索器初始化完成！")
    except Exception as e:
        logger.error(f"搜索器初始化失败: {e}", exc_info=True)
        raise e

    yield  # 让 FastAPI 处理请求

    # 退出时的清理工作
    logger.info("正在关闭服务：清理搜索器底层连接池...")
    if hasattr(app.state, "searcher") and hasattr(app.state.searcher, "http_session"):
        app.state.searcher.http_session.close()
    logger.info("资源清理完毕。")


# 挂载 lifespan 生命周期
app = FastAPI(title="法律数据库搜索API", lifespan=lifespan)


class SearchRequest(BaseModel):
    db_type: Literal["legal_provisions"]
    query: str
    top_k: int = 10
    score: float = 0.5


@app.post("/search")
def search_provisions(req: SearchRequest, request: Request):
    logger.info(f"收到检索请求: db_type={req.db_type}, query='{req.query}', top_k={req.top_k}")
    try:
        # 添加简单校验
        if req.db_type != "legal_provisions":
            logger.warning(f"拒绝请求: 暂不支持的数据库类型 '{req.db_type}'")
            raise ValueError(f"暂不支持的数据库类型: {req.db_type}")

        # 从 app.state 中提取已经初始化好的实例
        searcher: LegalProvisionSearcher = request.app.state.searcher

        # 执行搜索
        results = searcher.search(query=req.query, top_k=req.top_k, score=req.score)

        logger.info(f"检索成功，共返回 {len(results)} 条结果。")
        return {
            "code": 200,
            "msg": "success",
            "data": results
        }
    except Exception as e:
        # 记录详细错误日志，exc_info=True 会把完整的错误堆栈打印到日志中
        logger.error(f"检索接口发生异常: {e}", exc_info=True)
        # 将错误抛出给客户端，保持健壮性
        raise HTTPException(status_code=500, detail=str(e))