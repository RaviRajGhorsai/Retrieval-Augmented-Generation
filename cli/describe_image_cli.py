import argparse
from lib.llm.multi_model_search import describe_image


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    image_parser = subparsers.add_parser(
        "generate-query", help="Generate a search query from an image"
    )
    image_parser.add_argument(
        "--query",
        type=str,
        help="Optional guiding prompt",
    )

    image_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Input image (enter path of image)",
    )

    args = parser.parse_args()

    match args.command:
        case "generate-query":
            describe_image(args.image, args.query)

        case _:
            parser.print_help()

    # run evaluation logic here


if __name__ == "__main__":
    main()
