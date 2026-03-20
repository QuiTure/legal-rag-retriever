import sys
from utils.legal_provision_searcher import LegalProvisionSearcher


def main():
    print("=" * 60)
    print("      法律条文智能搜索系统")
    print("=" * 60)

    # 1. 获取文件路径并初始化搜索器
    json_path = input("请输入向量化JSON文件路径 (默认: database/legal_provisions/中华人民共和国刑法.json): ").strip()
    if not json_path:
        json_path = "database/legal_provisions/中华人民共和国刑法.json"

    try:
        print(f"正在加载数据并初始化搜索器: {json_path} ...")
        searcher = LegalProvisionSearcher(json_path=json_path)
        print("初始化成功！\n")
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)

    # 2. 交互式主循环
    while True:
        print("*" * 60)
        query = input("请输入搜索的法律问题 (输入 'q' 退出): ").strip()

        if query.lower() in ['q', 'quit', 'exit']:
            print("退出系统，再见！")
            break
        if not query:
            continue

        top_k_str = input("请输入需要返回的结果数量 (默认 5): ").strip()
        top_k = int(top_k_str) if top_k_str.isdigit() else 5

        print(f"\n正在搜索关于 '{query}' 的相关法条 (Top {top_k})，请稍候...\n")

        # 3. 调用封装好的搜索器
        results = searcher.search(query=query, top_k=top_k)

        # 4. 结构化输出结果
        if not results:
            print("未找到相关的法律条文。")
            continue

        print("=" * 50)
        print(f" 检索完毕，共找到 {len(results)} 条相关法条")
        print("=" * 50)

        for i, item in enumerate(results, 1):
            print(f"【匹配结果 {i}】")
            # 新增：输出相似度（保留四位小数）
            score = item.get('相似度', 0)
            print(f"相似度: {score:.4f}" if isinstance(score, float) else f"相似度: {score}")
            print(f"法条: {item.get('法条', '未知位置')}")
            print(f"内容: {item.get('内容', '无内容')}")
            print("-" * 50)


if __name__ == "__main__":
    main()