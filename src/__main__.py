import re

from numpy import argmax, array, full, inf

from llm_sdk import Small_LLM_Model

from .models import IdGroups, StandardDict
from .utils import build_final_prompt, read_files


class Decoder:
    """ Class which uses Small_LLM_Model for Constrained Decoding. """

    def __init__(self, llm: Small_LLM_Model):
        self.llm = llm

    def create_index_id_groups(self, token_ids: IdGroups) -> IdGroups:

        max_length: int = max(len(f) for f in token_ids)

        index_id_groups: IdGroups = [
            [] for _ in range(max_length)
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

    def extract_functions_arguments(
        self,
        function_name_ids: list[int],
        functions_def: StandardDict
    ) -> list[str]:
        function_arg: list[str] = []
        function_name = "fn" + self.llm.decode(function_name_ids)

        for function_def in functions_def:
            if function_def["name"] == function_name:
                function_arg = [key for key in function_def['parameters']]
                break

        return function_arg

    def force_text(self, ids: list[int], text: str) -> list[int]:
        ids.extend(self.llm.encode(text)[0])

    def print_ids(self, idList: list[int]) -> None:
        response: str = self.llm.decode(idList).splitlines()[-1]

        print(response)

    def predict_next_tokens(
        self,
        index_id_groups: IdGroups,
        final_prompt_ids: list[int],
        token_ids: IdGroups
    ) -> list[int]:
        ids: list[int] = []

        for position in range(len(index_id_groups)):
            next_token_id: int = self.generate_next_token_id(
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

    def execute_prompt(
        self,
        prompt: str,
        functions_def: StandardDict
    ) -> None:

        final_prompt: str = build_final_prompt(prompt, functions_def)
        final_prompt_ids: list[int] = (
            self.llm.encode(final_prompt).tolist()[0]
        )

        functions_token_ids: IdGroups = [
            self.llm.encode(definition['name']).tolist()[0][1:]
            for definition in functions_def
        ]

        functions_index_id_groups: IdGroups = (
            self.create_index_id_groups(functions_token_ids)
        )

        function_name_ids: list[int] = self.predict_next_tokens(
            functions_index_id_groups,
            final_prompt_ids,
            functions_token_ids
        )

        function_arg: list[str] = self.extract_functions_arguments(
            function_name_ids,
            functions_def
        )
        arguments_amount: int = len(function_arg)

        self.force_text(
            final_prompt_ids, f'", "parameters": {{ "{function_arg[0]}": '
        )

        extracted_words_from_prompt: list[list[str]] = (
            re.findall(r'\w+', prompt)
        )
        extracted_words_id_list: IdGroups = []

        for word in extracted_words_from_prompt:
            extracted_words_id_list.append(
                self.llm.encode(word).tolist()[0]
            )

        extracted_words_index_id_group: IdGroups = (
            self.create_index_id_groups(extracted_words_id_list)
        )

        for i in range(arguments_amount):
            self.predict_next_tokens(
                extracted_words_index_id_group,
                final_prompt_ids,
                extracted_words_id_list
            )

            if i + 1 < arguments_amount:
                self.force_text(
                    final_prompt_ids, f', "{function_arg[i + 1]}": '
                )

        self.force_text(final_prompt_ids, ' }')

        # Nalezy rowniez wyciagnac typ arugmentu, zeby wiedziec czy
        # przypisujemy mu nawiasy "" czy tez nie - string czy int

        # Pamiętaj, ze parametr moze sie skladac z kilku tokenów!

        self.print_ids(final_prompt_ids)


def main() -> None:
    input_data, functions_def, output = read_files()
    decoder = Decoder(Small_LLM_Model())

    for input in input_data:
        decoder.execute_prompt(input['prompt'], functions_def)


if __name__ == "__main__":
    main()
