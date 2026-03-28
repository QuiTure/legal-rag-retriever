import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from utils.legal_provision_searcher import LegalProvisionSearcher


def setup_logging():
    """配置全局日志输出（与 Server 端保持一致的轮转日志与格式）"""
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    # 定义统一的日志格式
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    # 文件处理器（按大小轮转，最大 10MB，保留 5 个备份）
    file_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "cli_app.log"),  # 命名为 cli_app.log 以区分 server 的 app.log
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(log_format)

    # 配置根日志记录器 (Root Logger)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )

    # 屏蔽 requests/urllib3/httpx 的底层 DEBUG/INFO 噪音
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _parse_int(value: str, default: int, min_value: int = 1) -> int:
    """安全解析整数输入"""
    try:
        v = int(value.strip())
        return v if v >= min_value else default
    except Exception:
        return default


def _parse_float(value: str, default: float) -> float:
    """安全解析浮点输入"""
    try:
        return float(value.strip())
    except Exception:
        return default


def main():
    setup_logging()

    print("=" * 68)
    print("      法律条文智能搜索系统 (向量组检索版：整句+子句 Max Recall)")
    print("=" * 68)

    # 1. 初始化搜索器
    try:
        print("\n正在初始化搜索器...")
        print("(系统会在首次查询时自动路由并加载相关法律文件)\n")
        searcher = LegalProvisionSearcher()
    except Exception as e:
        print(f"\n[错误] 初始化失败: {e}")
        sys.exit(1)

    # 2. 交互式主循环
    while True:
        try:
            print("\n" + "*" * 68)
            query = input("请输入搜索关键词 (支持多个，用中/英文逗号分割，输入 'q' 退出): ").strip()

            if query.lower() in ["q", "quit", "exit"]:
                print("退出系统，再见！")
                break
            if not query:
                continue

            # top_k: 最终返回数量
            top_k_str = input("请输入期望返回的最终结果数量 top_k (默认 5): ").strip()
            top_k = _parse_int(top_k_str, default=5, min_value=1)

            # retrieve_k: 初排召回数量（每个关键词）
            retrieve_k_str = input("请输入初排召回数量 retrieve_k (默认 50): ").strip()
            retrieve_k = _parse_int(retrieve_k_str, default=50, min_value=1)

            # score: 最低分阈值（通常是 rerank 分）
            score_str = input("请输入最低相似度阈值 score (默认 0): ").strip()
            score = _parse_float(score_str, default=0.0)

            print(
                f"\n正在分析请求 '{query}'，执行：LLM 路由 -> 向量组 Max 初排 -> 整句内容 Rerank...\n"
            )

            # 3. 调用新搜索器
            results = searcher.search(
                query=query,
                top_k=top_k,
                retrieve_k=retrieve_k,
                score=score
            )

            # 4. 输出结果
            if not results:
                print("\n未找到相关的法律条文。")
                continue

            print("\n" + "=" * 56)
            print(f" 检索完毕，共找到 {len(results)} 条相关法条 (按相似度降序)")
            print("=" * 56)

            for i, item in enumerate(results, 1):
                print(f"【匹配结果 {i}】")
                sim_score = item.get("相似度", 0)
                if isinstance(sim_score, (float, int)):
                    print(f"相似度: {sim_score:.4f}")
                else:
                    print(f"相似度: {sim_score}")

                print(f"法条名: {item.get('法条', '未知位置')}")
                print(f"内  容: {item.get('内容', '无内容')}")
                print("-" * 56)

        except KeyboardInterrupt:
            print("\n\n检测到中断信号，退出系统，再见！")
            break
        except Exception as e:
            print(f"\n[错误] 检索过程中发生异常: {e}")


if __name__ == "__main__":
    main()