import json
from argparse import ArgumentParser
from typing import NamedTuple

from numpy import argmax, array, full, inf

from llm_sdk import Small_LLM_Model

Paths = tuple[str, str, str]
Input = dict[str, str]
StandardDict = dict[str, str]


class VocabularyData(NamedTuple):
    input_data: StandardDict
    functions_def: StandardDict
    output: StandardDict
    vocab: dict[str, int]


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


def build_vocab_id_str(vocab: dict[str, int]) -> dict[int, str]:
    return {token_id: tok.replace("Ġ", " ") for tok, token_id in vocab.items()}


def execute_prompt(
    prompt: str,
    functions_def: StandardDict,
    llm_model: Small_LLM_Model,
    vocab: dict[str, int]
) -> None:
    # Testing
    reverse_vocab = build_vocab_id_str(vocab)

    prompt_ids: list[int] = llm_model.encode(prompt).tolist()[0]
    final_prompt: str = build_final_prompt(prompt, functions_def)
    final_prompt_ids: list[int] = llm_model.encode(final_prompt).tolist()[0]

    functions_token_ids: list[list[int]] = [
        llm_model.encode(definition['name']).tolist()[0][1:]
        for definition in functions_def
    ]
    max_function_length = max(len(f) for f in functions_token_ids)

    function_token_groups: list[list[int]] = [
        [] for _ in range(max_function_length)
    ]

    for function_ids in functions_token_ids:
        function_token_groups[0].append(function_ids[0])

    function_name_ids: list[int] = []
    for position in range(len(function_token_groups)):
        next_token_id: int = generate_next_token_id(
            final_prompt_ids,
            function_token_groups[position]
        )

        final_prompt_ids.append(next_token_id)
        function_name_ids.append(next_token_id)

        for index, function_ids in enumerate(functions_token_ids):
            if (
                position + 1 < len(function_ids)
                and next_token_id == function_ids[position]
            ):
                token_id = function_ids[position + 1]
                function_token_groups[position + 1].append(token_id)

        response: str = llm_model.decode(final_prompt_ids).splitlines()[-1]

        if (
            position + 1 >= len(function_token_groups)
            or function_token_groups[position + 1] == []
        ):
            break

    function_arg: list[str] = extract_functions_arguments(
        function_name_ids,
        llm_model
    )

    force_text(final_prompt_ids, f'", "parameters": {{ "{function_arg[0]}": "')

    next_token_id: int = generate_next_token_id(
        final_prompt_ids,
        prompt_ids
    )

    final_prompt_ids.append(next_token_id)

    if len(function_arg) == 1:
        force_text(final_prompt_ids, '" }}')
    else:
        force_text(final_prompt_ids, f'", {function_arg[1]}: "')

        next_token_id: int = generate_next_token_id(
            final_prompt_ids,
            prompt_ids
        )
        final_prompt_ids.append(next_token_id)

        force_text(final_prompt_ids, '" }')

    response: str = llm_model.decode(final_prompt_ids).splitlines()[-1]
    print(response)


def force_text(ids: list[int], text: str) -> list[int]:
    ids.extend(llm_model.encode(text)[0])


def mask_logits(loits: list[float], allowed_ids: list[int]) -> list[int]:
    arr = array(loits)
    masked = full(arr.shape, -inf)

    if allowed_ids:
        masked[allowed_ids] = arr[allowed_ids]

    return masked.tolist()


def generate_next_token_id(id_list: list[int], allowed_ids: list[int]) -> int:
    logits: list[float] = llm_model.get_logits_from_input_ids(
        id_list
    )
    masked_logits: list[int] = mask_logits(
        logits, allowed_ids
    )
    return int(argmax(masked_logits))


def parser() -> Paths:
    arg_parser = ArgumentParser(description="Call Me Maybe")

    arg_parser.add_argument("--functions_definition", required=True)
    arg_parser.add_argument("--input", required=True)
    arg_parser.add_argument("--output", required=True)

    args = arg_parser.parse_args()

    return (args.functions_definition, args.input, args.output)


def extract_functions_arguments(
    function_name_ids: list[int],
    llm_model: Small_LLM_Model
) -> list[str]:
    function_arg: list[str] = []
    function_name = "fn" + llm_model.decode(function_name_ids)

    for function_def in functions_def:
        if function_def["name"] == function_name:
            function_arg = [key for key in function_def['parameters']]
            break

    return function_arg


def read_files(vocab_path: str) -> VocabularyData:
    definitions_path, input_path, output_path = parser()

    with open(input_path, "r") as file:
        input_data = json.load(file)

    with open(definitions_path, "r") as file:
        functions_def = json.load(file)

    with open(vocab_path, "r") as file:
        vocab = json.load(file)

    output: dict[str, str] = {}

    return (input_data, functions_def, output, vocab)


if __name__ == "__main__":
    llm_model = Small_LLM_Model()
    vocab_path = llm_model.get_path_to_vocab_file()

    input_data, functions_def, output, vocab = read_files(vocab_path)

    for input in input_data:
        execute_prompt(input['prompt'], functions_def, llm_model, vocab)

    # execute_prompt(input_data[-1]['prompt'], functions_def, llm_model, vocab)
