#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify_model", help="Verify model initialized"
    )
    embed_parser = subparsers.add_parser(
        "embed_text", help="Embed text using embedding model"
    )

    embed_parser.add_argument("text", type=str, help="input text for embedding")

    document_embed_parser = subparsers.add_parser(
        "verify_embeddings", help="Generates embeddings of document"
    )

    query_embed_parse = subparsers.add_parser(
        "embedquery", help="Generates embedding of the user query"
    )
    query_embed_parse.add_argument("query", type=str, help="Input user query")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.query)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
