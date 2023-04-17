# incorrectness_cascade

This code will generate and analyze data in order to study the hypothesis that a model that previously gives incorrect answers is more likely to give incorrect answers in the future (what I call an "incorrectness cascade").

Study pre-registration: https://aizi.substack.com/p/pre-registering-a-study

To replicate this work:
1. Install this repo
2. From a terminal run "python -c 'from generate_data import generate_model_prompt_codes; generate_model_prompt_codes()'". This generates model_prompt_codes.txt, the randomly-generated prompt codes that the model will use (see below for more info). This only needs to be done once. Consider also changing the random seed used.
3. From a terminal run "python generate_data.py". This will read model_prompt_codes.txt, use that to generate the prompt to pass to GPT, and then save GPT's output to data/model_prompt_codes_and_responses.txt. (Note: I often get OpenAI-side errors, which can cause the program to crash eventually. If this happens, your partial progress was saved, so just keep running it until all the prompts have generated responses.) You will need to provide your own OpenAI API key. You can find instructions for setting that here: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety. 
4. Optionally, graph your data by running "python score_data.py". This will generate two line graphs in the figures/ directory.
5. From a terminal run "python run_statistics.py". This will run all six tests, plus some bonus tests. Outputs are printed to the terminal (the best interface). Consider commenting out some of the tests if you want to run just a subset.

Notes on prompt codes:

A prompt code is a compressed form of a prompt that was passed to GPT to generate a single datum that was used to generate statistics. Each model call is encoded this way:

```PP.XX.NNN.(KK[ab]){XX+1}/R```

- Each . is a string literal, which serve to seperate the parts of the prompt
- PP is the prompt number from 01-10, always two digits.
- XX is the number of false statementsfrom 00-10, always two digits.
- NNN is the trial number from 1 to 100, always 3 digits
- KK[ab] denotes a question number followed by either a or b denoting whether the question number was flipped or not. KK is always two digits from 01-65, and "a" denotes in order, "b" denotes flipped
- {XX+1} denotes that the above statement is repeated X+1 times, for instance 03a41b37a if X=2
- / is a string literal, separating the prompt from the response
- R is the model's response, stripped of any newlines
