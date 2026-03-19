import argparse
from lib.llm.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate search performance using Precision@k, Recall@k, and F1 score",
    )
    evaluate_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()

    match args.command:
        case "evaluate":
            evaluate(args.limit)

        case _:
            parser.print_help()

    # run evaluation logic here


if __name__ == "__main__":
    main()
