import matplotlib.pyplot as plt
import numpy as np
import re


def load_prompt_abbreviations():
    # short terms we use to refer to the prompts
    prompt_abbreviations=[
        "None",
        "Correctly",
        "Incorrectly",
        "HHH",
        "Consistently",
        "(Wa)luigi",
        "IQ 100",
        "IQ 150",
        "IQ 200",
        "IQ 1000",
    ]
    return prompt_abbreviations

def write_misformatted():
    # loops over all model answers in data/model_prompt_codes_and_responses.txt, and if it is misformatted, 
    # it writes that answer to data/misformatted_responses.txt 
    mistakes=[]
    with open("input_data/boolean_statements.txt") as f_questions:
        boolean_statements=f_questions.read().split("\n")
    with open('data/model_prompt_codes_and_responses.txt', 'r') as f_data:
        lines=f_data.read().split("\n")
    for line in lines:
        prompt_match=re.search(r"(\d{2}).\d{2}.\d{3}.((\d{2}[a|b]){0,11})/[\n]?", line)
        prompt_questions_code=prompt_match.group(2)
        final_prompt_question=int(prompt_questions_code[-3:-1])
        question=boolean_statements[final_prompt_question-1]
        correct_answer = re.search("[(](\w*)[/](\w*)[)]", question).group(1).lower()
        incorrect_answer = re.search("[(](\w*)[/](\w*)[)]", question).group(2).lower()
        full_AI_answer=line.split("/")[1]
        reduced_AI_answer=full_AI_answer.split(" ")[0].replace("\n","").replace(".", "").lower()
        if reduced_AI_answer==correct_answer or reduced_AI_answer==incorrect_answer:
            continue
        else:
            mistakes.append(line)
    with open("data/misformatted_responses.txt", "a") as f_out:
        f_out.write("\n".join(mistakes))

def count_categories():
    # loops over all model answers in data/model_prompt_codes_and_responses.txt, 
    # and classifies them as correct/incorrect/misformatted
    # returns a 4-10-by-11 list block, for total/correct/incorrect/misformatted   
    all_answers=[[0 for _ in range(11)] for __ in range(10)]
    correct_answers=[[0 for _ in range(11)] for __ in range(10)]
    incorrect_answers=[[0 for _ in range(11)] for __ in range(10)]
    misformatted_answers=[[0 for _ in range(11)] for __ in range(10)]
    with open("input_data/boolean_statements.txt") as f_questions:
        boolean_statements=f_questions.read().split("\n")
    with open('data/model_prompt_codes_and_responses.txt', 'r') as f_data:
        lines=f_data.read().split("\n")
    for line in lines:
        prompt_match=re.search(r"(\d{2}).(\d{2}).\d{3}.((\d{2}[a|b]){0,11})/[\n]?", line)
        PP=int(prompt_match.group(1))
        XX=int(prompt_match.group(2))
        prompt_questions_code=prompt_match.group(3)
        final_prompt_question=int(prompt_questions_code[-3:-1])
        question=boolean_statements[final_prompt_question-1]
        correct_answer = re.search("[(](\w*)[/](\w*)[)]", question).group(1).lower()
        incorrect_answer = re.search("[(](\w*)[/](\w*)[)]", question).group(2).lower()
        full_AI_answer=line.split("/")[1]
        reduced_AI_answer=full_AI_answer.split(" ")[0].replace("\n","").replace(".", "").lower()
        all_answers[PP-1][XX]+=1
        if reduced_AI_answer==correct_answer:
            correct_answers[PP-1][XX]+=1
        elif reduced_AI_answer==incorrect_answer:
            incorrect_answers[PP-1][XX]+=1
        else:
            misformatted_answers[PP-1][XX]+=1
    return [all_answers, correct_answers, incorrect_answers, misformatted_answers]
    

def plot_categories_heatmap():
    # makes a picture of which (X,P) values have a large or small population
    data=np.array(count_categories())
    data=data/data[0]
    data=data[1:4]
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    plt.suptitle("Chance of each type of AI answer")

    # Flatten the axs array to easily loop through it
    axs_flat = axs.flatten()

    titles=["Correct", "Incorrect", "Misformatted"]

    # Loop through each heatmap in data and plot it
    for idx, heatmap in enumerate(data):
        im = axs_flat[idx].imshow(heatmap, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        axs_flat[idx].set_title(titles[idx])
        axs_flat[idx].set_xlabel('# Previous Incorrect Answers')
        axs_flat[idx].set_ylabel('Prompt')


    # Add a colorbar for the heatmaps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Leave space for the colorbar
    plt.show()

def plot_categories_line_graph(omit_incorrect_prompt=False):
    # makes a line graph of y as a function of P and X
    data=np.array(count_categories())
    data=data[2]/(data[1]+data[2])
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    prompt_abbreviations=load_prompt_abbreviations()
    # Generate an X axis with 11 points (assuming the 10x11 array represents 11 points for each line)
    x = np.arange(11)

    # Loop through each row in data and plot it as a line graph
    for idx, row in enumerate(data):
        if omit_incorrect_prompt and idx==2:
            continue
        label=prompt_abbreviations[idx]
        ax.plot(x, row, label=label)


    # Customize the plot
    ax.set_title('Y as a function of X and P')
    ax.set_xlabel('X - Number of Previous Factually-Incorrect Answers')
    ax.set_ylabel('Y - Frequency of Incorrect Answer from GPT-3.5')
    ax.legend()

    # Save the plot
    figure_title="figures/y_x_p_line_graph.png"
    if omit_incorrect_prompt:
        figure_title="figures/y_x_p_line_graph_omit_2.png"

    plt.savefig(figure_title)

if __name__=="__main__":
    # count_categories()
    # write_misformatted()
    plot_categories_heatmap()
    plot_categories_line_graph()
    plot_categories_line_graph(omit_incorrect_prompt=True)
