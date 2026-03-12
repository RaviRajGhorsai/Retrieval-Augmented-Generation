#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_command,
    semantic_chunk_command,
    embed_chunks_command,
    search_chunked_command,
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

    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", type=str, help="Input user query")
    search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Returns required number of results only (default = 5)",
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk texts")
    chunk_parser.add_argument("text", type=str, help="Input text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=200,
        help="Specify chunk size, default=200",
    )
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap Chunking")

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Apply Chunking sematically, respecting natural language"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Input text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Chunk size"
    )
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="overlap")

    embed_chunk_parser = subparsers.add_parser(
        "embed_chunks", help="Embed the chunks for documents"
    )
    search_chunk_parser = subparsers.add_parser(
        "search_chunked", help="Semantic Search"
    )
    search_chunk_parser.add_argument("query", type=str, help="Inpur query")
    search_chunk_parser.add_argument("--limit", type=int, default=5, help="top n results")

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

        case "search":
            search_command(args.query, args.limit)

        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)

        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)

        case "embed_chunks":
            embed_chunks_command()
        
        case "search_chunked":
            search_chunked_command(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
