import argparse
from lib.llm.augmented_generation import rag
from lib.llm.summary import summary, answer_question


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser(
        "summary", help="AI Summary of the search results."
    )
    summary_parser.add_argument("query", type=str, help="Search query for RAG")
    summary_parser.add_argument(
        "--limit", type=int, default=5, help="Display top n results."
    )

    question_parser = subparsers.add_parser("question", help="Ask a question based on retrieved data")
    question_parser.add_argument("question", type=str, help="Ask a question based on retrieved data")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Display top n results."
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag(query)

        case "summary":
            summary(args.query, args.limit)

        case "question":
            answer_question(args.question, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
