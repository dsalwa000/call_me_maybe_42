from typing import NamedTuple

Paths = tuple[str, str, str]
Input = dict[str, str]
StandardDict = dict[str, str]
IdGroups = list[list[int]]


class VocabularyData(NamedTuple):
    input_data: StandardDict
    functions_def: StandardDict
    output: StandardDict
    vocab: dict[str, int]
