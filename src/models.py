from typing import NamedTuple, Literal

Paths = tuple[str, str, str]
Input = dict[str, str]
StandardDict = dict[str, str]
IdGroups = list[list[int]]
FunctionParameters = dict[str, str]
IndexGroup = list[list[int]]
IdList = list[int]
ArgType = Literal['string', 'int']


class VocabularyData(NamedTuple):
    input_data: StandardDict
    functions_def: StandardDict
    output: StandardDict
    vocab: dict[str, int]
