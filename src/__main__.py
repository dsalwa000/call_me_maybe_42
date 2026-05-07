from argparse import ArgumentParser
from typing import TypeAlias
import json

Paths: TypeAlias = tuple[str, str, str]
Input: TypeAlias = dict[str, str]


def parser() -> Paths:
    arg_parser = ArgumentParser(description="Call Me Maybe")

    arg_parser.add_argument("--functions_definition", required=True)
    arg_parser.add_argument("--input", required=True)
    arg_parser.add_argument("--output", required=True)

    args = arg_parser.parse_args()

    return (args.functions_definition, args.input, args.output)


def print_data(description: str, data: list[Input]) -> None:
    print(description)
    for log in data:
        print(log)
    print()


if __name__ == "__main__":
    definitions_path, input_path, output_path = parser()

    with open(definitions_path, "r") as file:
        definitions = json.load(file)

    with open(input_path, "r") as file:
        input = json.load(file)

    print_data("Definitions", definitions)
    print_data("Input", input)
