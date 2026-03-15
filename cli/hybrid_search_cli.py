import argparse
from lib.hybrid_search import (
    normalize_command,
    weighted_search_command,
    rrf_search_command,
)


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

    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Perform weighted search (combine both bm25 score and semantic score)",
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        help="""alpha (or "α") is just a constant that we can use to dynamically control the weighting between the two scores""",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, default=5, help="Top n results: default=5"
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search",
        help="Perform rrf search (combine both bm25 score and semantic score)",
    )
    rrf_search_parser.add_argument("query", type=str, help="User query")
    rrf_search_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="The k parameter (a constant) controls how much more weight we give to higher-ranked results vs. lower-ranked ones",
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, default=5, help="Returns top n results"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)

        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)

        case "rrf-search":
            rrf_search_command(args.query, args.limit, args.k)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
