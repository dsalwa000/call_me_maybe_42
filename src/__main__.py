import json
from argparse import ArgumentParser

from numpy import argmax

from llm_sdk import Small_LLM_Model

Paths = tuple[str, str, str]
Input = dict[str, str]


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
        f'{{"prompt": "{user_prompt}", "name": '
    )


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


def execute_prompts(input_data: list[str], functions_def: list[dict]):
    for input in input_data:
        MAX_NEW_TOKENS = 40
        i = 0
        user_prompt = input['prompt']

        final_prompt = build_final_prompt(user_prompt, functions_def)
        ids = llm_model.encode(final_prompt).tolist()[0]

        while i < MAX_NEW_TOKENS:
            logits = llm_model.get_logits_from_input_ids(ids)
            next_token_id = int(argmax(logits))
            ids.append(next_token_id)
            i += 1

        decoded = llm_model.decode(ids)
        print(f"\nFinal: {decoded}")


if __name__ == "__main__":
    llm_model = Small_LLM_Model()
    definitions_path, input_path, output_path = parser()
    vocab_path = llm_model.get_path_to_vocab_file()

    with open(input_path, "r") as file:
        input_data = json.load(file)

    with open(definitions_path, "r") as file:
        functions_def = json.load(file)

    with open(vocab_path, "r") as file:
        vocab = json.load(file)

    execute_prompts(input_data, functions_def)
