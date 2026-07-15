from datetime import datetime


def main() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("invest-agent is ready.")
    print(f"Started at: {now}")
    print("Next step: implement your LangGraph workflow under app/workflow/.")


if __name__ == "__main__":
    main()

