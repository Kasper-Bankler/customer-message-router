"""Command-line demo entry point: routes one customer message and prints the RoutingDecision as JSON. It holds no logic of its own: everything it prints comes from pipeline.route_message.

    python -m src.cli "I lost my card and someone is using it"
"""

import argparse

from src.pipeline import route_message


def main() -> None:
    parser = argparse.ArgumentParser(description="Route one customer message.")
    parser.add_argument("message", help="The customer message to route.")
    args = parser.parse_args()

    decision = route_message(args.message)
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
