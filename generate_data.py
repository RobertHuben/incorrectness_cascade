import random
import re
import openai
import os


def generate_model_prompt_codes():
    # generates the encoded version of the model prompts and saves them to model_prompt_codes.txt
    # see readme for the explanation of the encoding
    random.seed(0)
    out_file_name = "model_prompt_codes.txt"
    codes = []
    question_set = list(range(1, 66))
    for PP in range(1, 11):
        # loop over prompts 1-10
        for XX in range(0, 11):
            # look over X=0-10
            if XX == 0:
                # if X=0, there are no false answers so we query exactly once
                N_max = len(question_set)
            else:
                # otherwise we query 100 times
                N_max = 100
            for NNN in range(1, N_max+1):
                # loop 100 times (for X>0)
                if XX == 0:
                    question_codes = [question_set[NNN-1]]
                else:
                    question_codes = question_set.copy()
                    random.shuffle(question_codes)
                    question_codes = question_codes[0:XX+1]
                question_codes = [
                    f"{code:02}{random.choice(['a','b'])}" for code in question_codes]
                joined_question_codes = "".join(question_codes)
                model_code = f"{PP:02}.{XX:02}.{NNN:03}.{joined_question_codes}/"
                codes.append(model_code)
    to_write = "\n".join(codes)
    with open(out_file_name, 'w') as f:
        f.write(to_write)


def expand_prompt_code_to_messages(prompt_code):
    # takes in a prompt code as generated by model_prompt_codes, and returns the messages that will be passed to GPT for it
    messages = []
    prompt_match = re.search(
        r"(\d{2}).\d{2}.\d{3}.((\d{2}[a|b]){0,11})/[\n]?", prompt_code)
    prompt_supplement_number = int(prompt_match.group(1))
    with open("input_data/prompt_supplements.txt") as f:
        prompt_supplements = f.read().split("\n")
        prompt_supplement = prompt_supplements[prompt_supplement_number-1]
        full_system_message = f"You are a question-answering AI. {prompt_supplement}Select one of the two words in parentheses to complete the sentence. Your answer should be a single word, and you should not say anything besides your single word answer. Example: 'Question: Alice prefers (cats/dogs) Answer: cats'."
        messages.append({"role": "system", "content": full_system_message})
    prompt_questions_code = prompt_match.group(2)
    while prompt_questions_code:
        next_question_code = prompt_questions_code[0:3]
        prompt_questions_code = prompt_questions_code[3:]
        next_question_number = int(next_question_code[0:2])
        next_question_is_flipped = next_question_code[2] == "b"
        with open("input_data/boolean_statements.txt") as f:
            boolean_statements = f.read().split("\n")

            next_question = boolean_statements[next_question_number-1]
            if next_question_is_flipped:
                next_question = swap_choices_in_question(next_question)
            messages.append({"role": "user", "content": next_question})
            if prompt_questions_code:
                messages.append({"role": "assistant", "content": incorrect_answer(
                    next_question, next_question_is_flipped)})

    return messages


def swap_choices_in_question(question):
    # takes in a question as a string, and swaps the string
    match = re.search("([\w ,']*[(])(\w*)([/])(\w*)([)][\w ]*\n?)", question)
    swapped_question = "".join([match.group(i) for i in [1, 4, 3, 2, 5]])
    return swapped_question


def incorrect_answer(question, question_is_flipped):
    # gives the incorrect answer to the question
    # if question_is_flipped, the incorrect answer is first, otherwise it is the second choice
    if question_is_flipped:
        question_where_correct_choice_is_first = swap_choices_in_question(
            question)
    else:
        question_where_correct_choice_is_first = question
    incorrect_answer = re.search(
        "[(](\w*)[/](\w*)[)]", question_where_correct_choice_is_first).group(2)
    return incorrect_answer


def call_model_from_prompt_code(prompt_code):
    # calls GPT 3.5 using the prompt given by prompt_code, and returns the model's answer
    messages = expand_prompt_code_to_messages(prompt_code)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301", messages=messages, max_tokens=5, temperature=1)
    model_answer = completion["choices"][0]["message"]["content"]
    return model_answer


def generate_data():
    # loops over all prompt codes, calls GPT on them, and saves it to data/model_prompt_codes_and_responses.txt
    # saves as you go, and it will only call a prompt code if it doesn't already have an answer recorded
    # you need to set your API key in your local environment, you can find instructions for that here: https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
    openai.api_key = os.environ["OPENAI_API_KEY"]
    with open('model_prompt_codes.txt', 'r') as f_in:
        queries = f_in.read().split("\n")
    with open('data/model_prompt_codes_and_responses.txt', 'r') as f_current:
        current_text = f_current.read()
    for query in queries:
        if query in current_text:
            continue
        else:
            model_response = call_model_from_prompt_code(query)
            with open('data/model_prompt_codes_and_responses.txt', 'a') as f_out:
                f_out.write(f"{query}{model_response}\n")
            print(f"Just wrote prompt code {query}")


if __name__ == "__main__":
    # generate_model_prompt_codes()
    # generate_data()

    for _ in range(100):
        try:
            generate_data()
        except:
            continue