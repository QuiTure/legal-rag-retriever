import sys
import logging
from utils.legal_provision_searcher import LegalProvisionSearcher


def setup_logging():
    """配置基础日志输出，以便查看底层搜索器的加载状态"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # 屏蔽 requests/urllib3 的底层 DEBUG/INFO 噪音
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    setup_logging()

    print("=" * 60)
    print("      法律条文智能搜索系统 (支持多关键词组合检索)")
    print("=" * 60)

    # 1. 获取文件路径并初始化搜索器
    default_path = "database/legal_provisions/中华人民共和国刑法.json"
    json_path = input(f"请输入向量化JSON文件路径 (默认: {default_path}): ").strip()
    if not json_path:
        json_path = default_path

    try:
        print(f"\n正在加载数据并初始化搜索器 (构建向量矩阵加速中): {json_path} ...")
        searcher = LegalProvisionSearcher(json_path=json_path)
        print("初始化成功！\n")
    except Exception as e:
        print(f"\n[错误] 初始化失败: {e}")
        sys.exit(1)

    # 2. 交互式主循环
    while True:
        try:
            print("*" * 60)
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

            print(f"\n正在快速检索 '{query}'，进行批量并发重排中，请稍候...\n")

            # 3. 调用封装好的搜索器
            results = searcher.search(query=query, top_k=top_k)

            # 4. 结构化输出结果
            if not results:
                print("未找到相关的法律条文。")
                continue

            print("=" * 50)
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