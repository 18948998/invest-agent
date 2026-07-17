"""invest-agent 主入口 —— 启动对话式投资研究助手。"""

from __future__ import annotations

from app.agents.main_agent import MainAgent


def main() -> None:
    agent = MainAgent()
    agent.run()


if __name__ == "__main__":
    main()
