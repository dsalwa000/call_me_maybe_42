import json
import re
from argparse import ArgumentParser
from typing import NamedTuple

from numpy import argmax, array, full, inf

from llm_sdk import Small_LLM_Model

Paths = tuple[str, str, str]
Input = dict[str, str]
StandardDict = dict[str, str]
IdGroups = list[list[int]]


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


def create_index_id_groups(
    token_ids: IdGroups
) -> IdGroups:

    max_length: int = max(len(f) for f in token_ids)

    index_id_groups: IdGroups = [
        [] for _ in range(max_length)
    ]

    index_id_groups[0].extend(ids[0] for ids in token_ids)
    return index_id_groups


def execute_prompt(
    prompt: str,
    functions_def: StandardDict,
    llm_model: Small_LLM_Model,
    vocab: dict[str, int]
) -> None:
    # Testing
    reverse_vocab = build_vocab_id_str(vocab)

    final_prompt: str = build_final_prompt(prompt, functions_def)
    final_prompt_ids: list[int] = llm_model.encode(final_prompt).tolist()[0]

    functions_token_ids: IdGroups = [
        llm_model.encode(definition['name']).tolist()[0][1:]
        for definition in functions_def
    ]

    functions_index_id_groups: IdGroups = (
        create_index_id_groups(functions_token_ids)
    )

    function_name_ids: list[int] = predict_next_tokens(
        functions_index_id_groups,
        final_prompt_ids,
        functions_token_ids
    )

    function_arg: list[str] = extract_functions_arguments(
        function_name_ids,
        llm_model
    )
    arguments_amount: int = len(function_arg)

    force_text(final_prompt_ids, f'", "parameters": {{ "{function_arg[0]}": ')

    extracted_words_from_prompt: list[list[str]] = re.findall(r'\w+', prompt)
    extracted_words_id_list: IdGroups = []

    for word in extracted_words_from_prompt:
        extracted_words_id_list.append(llm_model.encode(word).tolist()[0])

    extracted_words_index_id_group: IdGroups = (
        create_index_id_groups(extracted_words_id_list)
    )

    for i in range(arguments_amount):
        predict_next_tokens(
            extracted_words_index_id_group,
            final_prompt_ids,
            extracted_words_id_list
        )

        if i + 1 < arguments_amount:
            force_text(
                final_prompt_ids, f', "{function_arg[i + 1]}": '
            )

    force_text(final_prompt_ids, ' }')

    # Nalezy rowniez wyciagnac typ arugmentu, zeby wiedziec czy przypisujemy mu
    # nawiasy "" czy tez nie - string czy int

    # Pamiętaj, ze parametr moze sie skladac z kilku tokenow!

    print_ids(final_prompt_ids)


def print_ids(idList: list[int]) -> None:
    response: str = llm_model.decode(idList).splitlines()[-1]

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


def predict_next_tokens(
    index_id_groups: IdGroups,
    final_prompt_ids: list[int],
    token_ids: IdGroups
) -> list[int]:
    ids: list[int] = []

    for position in range(len(index_id_groups)):
        next_token_id: int = generate_next_token_id(
            final_prompt_ids,
            index_id_groups[position]
        )

        final_prompt_ids.append(next_token_id)
        ids.append(next_token_id)

        for function_ids in token_ids:
            if (
                position + 1 < len(function_ids)
                and next_token_id == function_ids[position]
            ):
                token_id = function_ids[position + 1]
                index_id_groups[position + 1].append(token_id)

        if (
            position + 1 >= len(index_id_groups)
            or index_id_groups[position + 1] == []
        ):
            break

    return ids


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

    # execute_prompt(input_data[1]['prompt'], functions_def, llm_model, vocab)
