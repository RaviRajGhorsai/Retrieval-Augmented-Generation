import argparse
from lib.llm.multi_model_search import describe_image


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--query",
        type=str,
        help="Input query",
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Input image (enter path of image)",
    )

    args = parser.parse_args()

    describe_image(args.image, args.query)

    # run evaluation logic here


if __name__ == "__main__":
    main()
