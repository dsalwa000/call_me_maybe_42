from argparse import ArgumentParser
from llm_sdk import Small_LLM_Model
import json
import numpy as np

Paths = tuple[str, str, str]
Input = dict[str, str]


def parser() -> Paths:
    arg_parser = ArgumentParser(description="Call Me Maybe")

    arg_parser.add_argument("--functions_definition", required=True)
    arg_parser.add_argument("--input", required=True)
    arg_parser.add_argument("--output", required=True)

    args = arg_parser.parse_args()

    return (args.functions_definition, args.input, args.output)


def print_input(description: str, data: list[Input]) -> None:
    print(description)
    for log in data:
        print(log['prompt'])
    print()


def print_functions(description: str, data: list[Input]) -> None:
    print(description)
    for log in data:
        print(log["name"])
        print(log["description"])
        print()


if __name__ == "__main__":
    llm_model = Small_LLM_Model()
    definitions_path, input_path, output_path = parser()
    vocab_path = llm_model.get_path_to_vocab_file()

    with open(input_path, "r") as file:
        input_data = json.load(file)

    with open(definitions_path, "r") as file:
        definitions = json.load(file)

    with open(vocab_path, "r") as file:
        vocab = json.load(file)

    print_input("Prompts", input_data)
    print_functions("Prompts", definitions)

    MAX_NEW_TOKENS = 50
    i = 0
    prompt = input_data[3]['prompt']
    ids = llm_model.encode(prompt).tolist()[0]

    print(f"Prompt: {prompt}")
    while i < MAX_NEW_TOKENS:
        logits = llm_model.get_logits_from_input_ids(ids)
        next_token_id = int(np.argmax(logits))
        ids.append(next_token_id)
        i += 1

    decoded = llm_model.decode(ids)
    print(f"\nFinal: {decoded}")
