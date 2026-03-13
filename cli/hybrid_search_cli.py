import argparse
from lib.hybrid_search import normalize_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalization_parser = subparsers.add_parser(
        "normalize",
        help="Normalize scores (semantic or keyword score, i.e, BM25 score)",
    )
    normalization_parser.add_argument(
        "scores", type=float, nargs="+", help="List of scores to normalize"
    )

    args = parser.parse_args()

    match args.command:

        case "normalize":
            normalize_command(args.scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
