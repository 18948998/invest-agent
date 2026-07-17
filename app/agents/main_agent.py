"""主对话 agent —— 用户所有交互的唯一入口。"""

from __future__ import annotations


class MainAgent:
    """对话中枢：打招呼 → 理解意图 → 路由到对应工作流。

    用户启动后只跟这个 agent 对话，不需要命令行参数。
    """

    def greet(self) -> None:
        """打印欢迎信息，告知用户可用的功能。"""
        print("=" * 50)
        print("  您好！我是您的投资研究助手。")
        print("  目前我可以帮您做两件事：")
        print()
        print("  1. 推荐股票 —— 按策略从全市场筛选符合格雷厄姆标准的股票")
        print("  2. 分析股票 —— 深度分析某只股票的基本面")
        print()
        print("  您想做什么？（输入 1 或 2，输入 q 退出）")
        print("=" * 50)

    def run(self) -> None:
        """启动对话循环。"""
        self.greet()

        while True:
            choice = input("\n> ").strip()

            if choice == "q":
                print("再见！")
                break

            if choice == "1":
                print("\n好的，我来帮您筛选推荐股票。")
                print("（筛选流程尚未实现，这里先占位）")
                # TODO: 后续这里会调 self.run_screen()
                break

            elif choice == "2":
                print("\n好的，请告诉我您想分析哪只股票。")
                symbol = input("\n请输入股票代码（如 600519）：").strip()
                if symbol:
                    print(f"\n收到，我来分析 {symbol} 的基本面。")
                    print("（分析流程尚未实现，这里先占位）")
                    # TODO: 后续这里会调 self.run_analyze(symbol)
                    break
                else:
                    print("股票代码不能为空，请重新输入。")
            else:
                print("请输入 1、2 或 q。")


# 入口统一在 app/main.py，不要单独运行此文件
