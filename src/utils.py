
import json
from argparse import ArgumentParser

from .models import Paths, VocabularyData


def build_final_prompt(user_prompt: str, functions_def: list[dict]) -> str:
    functions_str = json.dumps(functions_def, indent=2)

    return (
        "You are a function calling assistant.\n"
        "The goal is to connect prompt with function name and specific "
        "parameters connected to the function. "
        'The final output should be a JSON with three parameters: "prompt", '
        '"name", "parameters".\n\n'
        "Available Functions:\n"
        f"{functions_str}\n\n"
        "Rules:\n"
        "- Return ONLY a valid JSON object.\n"
        '- The JSON object must contain three keys: "prompt", "name", and '
        '"parameters".\n'
        '- "prompt" must match the user input string.\n'
        '- "name" must be one of the available function names.\n'
        '- "parameters" must contain key-value pairs matching the expected '
        "function argument types.\n\n"
        "User Prompt:\n"
        f"{user_prompt}\n\n"
        "Response:\n"
        f'{{"prompt": "{user_prompt}", "name": "fn'
    )


def parser() -> Paths:
    arg_parser = ArgumentParser(description="Call Me Maybe")

    arg_parser.add_argument("--functions_definition", required=True)
    arg_parser.add_argument("--input", required=True)
    arg_parser.add_argument("--output", required=True)

    args = arg_parser.parse_args()

    return (args.functions_definition, args.input, args.output)


def read_files() -> VocabularyData:
    definitions_path, input_path, output_path = parser()

    with open(input_path, "r") as file:
        input_data = json.load(file)

    with open(definitions_path, "r") as file:
        functions_def = json.load(file)

    output: dict[str, str] = {}

    return (input_data, functions_def, output)
