import argparse
from lib.multimodal_search import verify_embedding, image_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    image_embedding = subparsers.add_parser(
        "verify-image-embedding", help="Verify image embeddings"
    )
    image_embedding.add_argument("image", type=str, help="Input image path")

    image_search = subparsers.add_parser(
        "image-search", help="Search movies based on image"
    )
    image_search.add_argument("img_path", type=str, help="Input path to image")

    args = parser.parse_args()

    match args.command:
        case "verify-image-embedding":
            verify_embedding(args.image)

        case "image-search":
            image_search_command(args.img_path)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
