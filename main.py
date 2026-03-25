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

    # 屏蔽 requests/urllib3/httpx 的底层 DEBUG/INFO 噪音 (LangChain 底层使用 httpx)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main():
    setup_logging()

    print("=" * 60)
    print("      法律条文智能搜索系统 (支持多关键词组合检索)")
    print("=" * 60)

    # 1. 初始化搜索器
    try:
        print("\n正在初始化搜索器...")
        print("(系统将在您首次查询时，自动调用大模型从本地法条库寻找并加载适用的法律文件)\n")
        searcher = LegalProvisionSearcher()
    except Exception as e:
        print(f"\n[错误] 初始化失败: {e}")
        sys.exit(1)

    # 2. 交互式主循环
    while True:
        try:
            print("\n" + "*" * 60)
            # 提示用户可以输入多个用逗号分割的关键词
            query = input("请输入搜索关键词 (支持多个，用中/英文逗号分割，输入 'q' 退出): ").strip()

            if query.lower() in ['q', 'quit', 'exit']:
                print("退出系统，再见！")
                break
            if not query:
                continue

            # 提示说明这是最终合并后返回的总数量上限
            top_k_str = input("请输入期望返回的最终结果数量 (默认 5): ").strip()
            top_k = int(top_k_str) if top_k_str.isdigit() else 5

            print(f"\n正在分析请求 '{query}'，进行动态路由、快速检索与并发重排，请稍候...\n")

            # 3. 调用封装好的搜索器
            results = searcher.search(query=query, top_k=top_k, score=0)

            # 4. 结构化输出结果
            if not results:
                print("\n未找到相关的法律条文。")
                continue

            print("\n" + "=" * 50)
            print(f" 检索完毕，共找到 {len(results)} 条相关法条 (按相似度降序排列)")
            print("=" * 50)

            for i, item in enumerate(results, 1):
                print(f"【匹配结果 {i}】")
                score = item.get('相似度', 0)
                # 格式化相似度保留4位小数
                if isinstance(score, (float, int)):
                    print(f"相似度: {score:.4f}")
                else:
                    print(f"相似度: {score}")

                print(f"法条名: {item.get('法条', '未知位置')}")
                print(f"内  容: {item.get('内容', '无内容')}")
                print("-" * 50)

        except KeyboardInterrupt:
            # 捕获 Ctrl+C，优雅退出
            print("\n\n检测到中断信号，退出系统，再见！")
            break
        except Exception as e:
            # 捕获搜索过程中的意外错误，防止整个程序崩溃退出
            print(f"\n[错误] 检索过程中发生异常: {e}")


if __name__ == "__main__":
    main()