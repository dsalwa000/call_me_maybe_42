import re

from numpy import argmax, array, full, inf

from llm_sdk import Small_LLM_Model

from .models import (
    FunctionParameters,
    IdGroups,
    IdList,
    IndexGroup,
    StandardDict,
    ArgType
)
from .utils import build_final_prompt, read_files


class Decoder:
    """ Class which uses Small_LLM_Model for Constrained Decoding. """

    def __init__(self, llm: Small_LLM_Model):
        self.llm: Small_LLM_Model = llm

        self.int_sufixes = [
            self.llm.encode(sufix).tolist()[0][0] for sufix in [',', '}']
        ]

        self.str_bracket = self.llm.encode('"').tolist()[0][0]

    def create_index_id_groups(self, token_ids: IdGroups) -> IdGroups:

        max_length: int = max(len(f) for f in token_ids)

        index_id_groups: IdGroups = [
            [] for _ in range(max_length + 1)
        ]

        index_id_groups[0].extend(ids[0] for ids in token_ids)
        return index_id_groups

    def mask_logits(
        self,
        loits: list[float],
        allowed_ids: list[int]
    ) -> list[int]:
        arr = array(loits)
        masked = full(arr.shape, -inf)

        if allowed_ids:
            masked[allowed_ids] = arr[allowed_ids]

        return masked.tolist()

    def generate_next_token_id(
        self,
        id_list: list[int],
        allowed_ids: list[int]
    ) -> int:
        logits: list[float] = self.llm.get_logits_from_input_ids(
            id_list
        )
        masked_logits: list[int] = self.mask_logits(
            logits, allowed_ids
        )
        return int(argmax(masked_logits))

    def extract_functions_parameters(
        self,
        function_name_ids: list[int],
        functions_def: StandardDict
    ) -> FunctionParameters:
        function_parameters: FunctionParameters = {}
        function_name = "fn" + self.llm.decode(function_name_ids[:-1])

        for function_def in functions_def:
            if function_def["name"] == function_name:
                function_parameters = {
                    key: value['type']
                    for key, value in function_def['parameters'].items()
                }
                break

        return function_parameters

    def force_text(self, ids: list[int], text: str) -> list[int]:
        ids.extend(self.llm.encode(text)[0])

    def print_ids(self, idList: list[int]) -> None:
        response: str = self.llm.decode(idList).splitlines()[-1]

        print(response)

    def extract_candidates(self, prompt: str) -> list[str]:
        quoted = re.findall(
            r'"([^"]*)"|\'([^\']*)\'', prompt
        )
        spans = [q for pair in quoted for q in pair if q]
        words = re.findall(r'\w+', prompt)
        return spans + words

    def extract_numbers(self, prompt: str) -> list[str]:
        return [w for w in re.findall(r'\d+(?:\.\d+)?', prompt)]

    def predict_next_tokens(
        self,
        possible_text_ids: IdGroups,
        sorted_index_groups: IndexGroup,
        final_prompt_ids: IdList,
        argType: ArgType
    ) -> IdList:
        ids: list[int] = []

        print(possible_text_ids)

        for position in range(len(sorted_index_groups)):

            next_token_id: int = self.generate_next_token_id(
                final_prompt_ids,
                sorted_index_groups[position]
            )

            final_prompt_ids.append(next_token_id)
            ids.append(next_token_id)

            possible_text_ids = [
                function_ids for function_ids in possible_text_ids
                if len(function_ids) > position
                and function_ids[position] == next_token_id
            ]

            for function_ids in possible_text_ids:
                if position + 1 == len(function_ids):
                    if argType == 'string':
                        sorted_index_groups[position + 1].append(
                            self.str_bracket
                        )
                    elif argType == 'int':
                        sorted_index_groups[position + 1].extend(
                            self.int_sufixes
                        )

                if position + 1 < len(function_ids):
                    token_id = function_ids[position + 1]
                    sorted_index_groups[position + 1].append(token_id)

            if (
                position + 1 >= len(sorted_index_groups)
                or sorted_index_groups[position + 1] == []
            ):
                break

        return ids

    def execute_prompt(
        self,
        prompt: str,
        functions_def: StandardDict
    ) -> None:

        final_prompt: str = build_final_prompt(prompt, functions_def)
        final_prompt_ids: list[int] = (
            self.llm.encode(final_prompt).tolist()[0]
        )

        function_names_ids: IdGroups = [
            self.llm.encode(definition['name']).tolist()[0][1:]
            for definition in functions_def
        ]

        function_names_index_groups: IdGroups = (
            self.create_index_id_groups(function_names_ids)
        )

        # Function name prediction
        function_name_ids: list[int] = self.predict_next_tokens(
            function_names_ids,
            function_names_index_groups,
            final_prompt_ids,
            'string'
        )

        # Parameters predictions
        self.force_text(
            final_prompt_ids, ', "parameters": { '
        )

        function_parameters: FunctionParameters = (
            self.extract_functions_parameters(
                function_name_ids,
                functions_def
            )
        )
        arguments_amount: int = len(function_parameters)

        print(self.extract_candidates(prompt))
        print(self.extract_numbers(prompt))

        for argument, arg_type in function_parameters.items():
            self.force_text(final_prompt_ids, f'"{argument}": ')

            if arg_type == 'string':
                self.force_text(final_prompt_ids, '"')

        self.print_ids(final_prompt_ids)


def main() -> None:
    input_data, functions_def, output = read_files()
    decoder = Decoder(Small_LLM_Model())

    # for input in input_data:
    #     decoder.execute_prompt(input['prompt'], functions_def)

    decoder.execute_prompt(input_data[-3]['prompt'], functions_def)


if __name__ == "__main__":
    main()
