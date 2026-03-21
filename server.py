import logging
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from utils.legal_provision_searcher import LegalProvisionSearcher

# 配置当前模块的 logger
logger = logging.getLogger(__name__)


# 使用 lifespan 替代全局初始化，确保跨 worker 的进程安全并管理资源
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在启动服务：初始化法律条文搜索器...")
    # 将实例化后的 searcher 挂载到 app.state 上，实现全局单例，避免多次连接外网
    app.state.searcher = LegalProvisionSearcher()
    logger.info("搜索器初始化完成！")

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
    try:
        # 添加简单校验
        if req.db_type != "legal_provisions":
            raise ValueError(f"暂不支持的数据库类型: {req.db_type}")

        # 从 app.state 中提取已经初始化好的实例
        searcher: LegalProvisionSearcher = request.app.state.searcher

        # 执行搜索
        results = searcher.search(query=req.query, top_k=req.top_k, score=req.score)

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