Model analizuje zapytania od uzytownika i nakierowuje wykonanie odpowiedniej
funkcji z listy functions_definitions.py

Zadanie polega na nakierowaiu LLM na zrozumienie intencji uzytkownika.

Workflow:
- Do modelu trafia prompt z _function_calling_tests.json_
- Wywołujesz LLM by dostać logity dla następnego znaku.
- LLM analizuje prompt i musi wybrać odpowiedni obiekt z _functions_definition.json_
    - Pierw wybieramy "name"
    - Z requestu wyławiamy odpowiednie parametry
- Maszyna stanów działa w trakcie generowania kazdego pojedyńczego tokenu
