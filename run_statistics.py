import numpy as np
from score_data import count_categories, load_prompt_abbreviations
import matplotlib.pyplot as plt
import scipy.stats as stats
import pymannkendall as mk
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd


def load_y():
    # loads the y data
    data = np.array(count_categories())
    YY = data[2]/(data[1]+data[2])
    return YY


def load_n():
    # loads the number of questions answered correctly/incorrectly (as opposed to misformatted)
    data = np.array(count_categories())
    NN = data[1]+data[2]
    return NN


def preliminary_test_most_misformatted():
    # says the highest concentration of misformatted answers
    data = np.array(count_categories())
    percent_misformatted = data[3]/data[0]
    print(np.max(percent_misformatted))


def test_1():
    # "Across each prompt P, compute the correlation coefficient between Y and X."
    y_data = load_y()
    x_data = np.array(range(11))
    prompt_abbreviations = load_prompt_abbreviations()

    correlations = []
    confidence_intervals = []
    for p_idx in range(y_data.shape[0]):
        y_p = y_data[p_idx, :]
        # Get the correlation coefficient between x_data and y_p
        corr_coef = np.corrcoef(x_data, y_p)[0, 1]

        # Compute the 95% confidence interval using Fisher's z-transformation
        z = np.arctanh(corr_coef)
        z_std_error = 1 / np.sqrt(y_p.size - 3)
        z_lower = z - 1.96 * z_std_error
        z_upper = z + 1.96 * z_std_error
        lower_bound = np.tanh(z_lower)
        upper_bound = np.tanh(z_upper)
        confidence_interval = (lower_bound, upper_bound)

        correlations.append(corr_coef)
        confidence_intervals.append(confidence_interval)

    # Print the correlation coefficients and their 95% confidence intervals for each prompt
    print("Correlation coefficients for each prompt:")
    for p_idx, (corr_coef, conf_interval) in enumerate(zip(correlations, confidence_intervals)):
        print(
            f"Prompt {prompt_abbreviations[p_idx]}: {corr_coef:.4f} (95% CI: {conf_interval[0]:.4f}, {conf_interval[1]:.4f})")
    graph_test_1(correlations, confidence_intervals)


def test_2():
    # "Across each prompt P, perform the Mann-Kendall test to see if Y is increasing as X ranges from 0 to 10."
    y_data = load_y()
    y_data = load_y()

    for p_idx in range(y_data.shape[0]):
        y_p = y_data[p_idx, :]
        mk_result = mk.original_test(y_p)

        # Print the results of the Mann-Kendall test for each prompt
        print(f"Mann-Kendall test for prompt {p_idx}: trend={mk_result.trend}, "
              f"slope={mk_result.slope:.3f}, p-value={mk_result.p:.4f}")


def test_3():
    # "Across each prompt P, perform the two-sample student's t-test comparing X=0 and X=1."
    correct_answers = np.array(count_categories())[1]
    incorrect_answers = np.array(count_categories())[2]
    x_first = 0
    x_second = 1

    for p_idx in range(correct_answers.shape[0]):
        y_first = np.concatenate((np.ones(correct_answers[p_idx, x_first]), np.zeros(
            incorrect_answers[p_idx, x_first])))
        y_second = np.concatenate((np.ones(correct_answers[p_idx, x_second]), np.zeros(
            incorrect_answers[p_idx, x_second])))
        ttest_result = stats.ttest_ind(y_first, y_second)

        # Print the results of the two-sample Student's t-test for each prompt
        print(
            f"Two-sample Student's t-test comparing X={x_first} and X={x_second} for prompt {p_idx+1}: statistic={ttest_result.statistic:.2f}, p-value={ttest_result.pvalue:.4f}")


def test_4():
    # "Across each prompt P, perform the two-sample student's t-test comparing X=1 and X=10."
    correct_answers = np.array(count_categories())[1]
    incorrect_answers = np.array(count_categories())[2]
    x_first = 1
    x_second = 10

    for p_idx in range(correct_answers.shape[0]):
        y_first = np.concatenate((np.ones(correct_answers[p_idx, x_first]), np.zeros(
            incorrect_answers[p_idx, x_first])))
        y_second = np.concatenate((np.ones(correct_answers[p_idx, x_second]), np.zeros(
            incorrect_answers[p_idx, x_second])))
        ttest_result = stats.ttest_ind(y_first, y_second)

        # Print the results of the two-sample Student's t-test for each prompt
        print(
            f"Two-sample Student's t-test comparing X={x_first} and X={x_second} for prompt {p_idx+1}: statistic={ttest_result.statistic:.2f}, p-value={ttest_result.pvalue:.4f}")


def test_5():
    # "Across each prompt P, perform the two-sample student's t-test comparing X=0 and X=10."
    correct_answers = np.array(count_categories())[1]
    incorrect_answers = np.array(count_categories())[2]
    x_first = 0
    x_second = 10

    for p_idx in range(correct_answers.shape[0]):
        y_first = np.concatenate((np.ones(correct_answers[p_idx, x_first]), np.zeros(
            incorrect_answers[p_idx, x_first])))
        y_second = np.concatenate((np.ones(correct_answers[p_idx, x_second]), np.zeros(
            incorrect_answers[p_idx, x_second])))
        ttest_result = stats.ttest_ind(y_first, y_second)

        # Print the results of the two-sample Student's t-test for each prompt
        print(
            f"Two-sample Student's t-test comparing X={x_first} and X={x_second} for prompt {p_idx+1}: statistic={ttest_result.statistic:.2f}, p-value={ttest_result.pvalue:.4f}")


def test_6():
    # "Perform a multiple-regression analysis of Y on X and (dummy-coded values of) P, with interaction terms between X and P. For this analysis, we will only consider P taking the values g-j. In particular, we will look for statistically significant interaction terms between X and P."
    y_data = load_y()

    # Select the relevant P values (6, 7, 8, and 9)
    y_data = y_data[6:10, :]

    # Create a DataFrame for the independent variables (X, dummy-coded P values, and interaction terms)
    X = np.arange(11)
    P = [6, 7, 8, 9]
    P_dummies = pd.get_dummies(P, drop_first=True)

    # Create a dataset for the analysis
    data = pd.DataFrame({'Y': y_data.flatten(), 'X': np.tile(X, len(P))})
    data = pd.concat([data, P_dummies.iloc[np.repeat(
        np.arange(len(P_dummies)), len(X))].reset_index(drop=True)], axis=1)

    # Rename columns
    data.columns = ['Y', 'X', 'P_7', 'P_8', 'P_9']

    # Create interaction terms
    data['X_P_7'] = data['X'] * data['P_7']
    data['X_P_8'] = data['X'] * data['P_8']
    data['X_P_9'] = data['X'] * data['P_9']

    # Perform the multiple regression analysis
    model = smf.ols(
        'Y ~ X + P_7 + P_8 + P_9 + X_P_7 + X_P_8 + X_P_9', data=data).fit()

    # Print the results
    print(model.summary())


def graph_test_1(correlations, confidence_intervals):
    # creates a graph showing the results of test 1
    prompts = np.arange(len(correlations))
    prompt_abbreviations = load_prompt_abbreviations()

    # Plot the correlation coefficients with their confidence intervals
    fig, ax = plt.subplots()
    ax.scatter(correlations, prompts, label='Correlation Coefficients')

    # Add error bars for the confidence intervals
    xerr = np.array([[corr - lower, upper - corr] for (lower, upper),
                    corr in zip(confidence_intervals, correlations)]).T
    ax.errorbar(correlations, prompts, xerr=xerr, fmt='o',
                capsize=5, label='95% Confidence Intervals')

    # Add a vertical dotted line at x=0
    ax.axvline(0, linestyle='--', color='gray', alpha=0.6)

    # Set the y ticks to the prompt abbreviations
    ax.set_yticks(prompts)
    ax.set_yticklabels(prompt_abbreviations)

    # Invert the y-axis
    ax.invert_yaxis()

    # Add labels and legends
    ax.set_ylabel('Prompt')
    ax.set_xlabel('Correlation between X and Y')
    ax.set_title('X-Y Correlation Coefficients with 95% Confidence Intervals')
    ax.legend()

    # Show the plot
    plt.show()


def test_bonus_1():
    # extra test not in the pre-registration, finding the coefficient on a linear regression of Y on X, for each value of P
    y_data = load_y()
    prompt_abbreviations = load_prompt_abbreviations()

    for p_idx in range(10):
        # Create a DataFrame for the variables (X and Y)
        X = np.arange(11)
        data = pd.DataFrame({'Y': y_data[p_idx, :].flatten(), 'X': X})

        # Perform the multiple regression analysis
        model = smf.ols('Y ~ X', data=data).fit()

        # Print the results
        print(
            f"X Coefficient for prompt {prompt_abbreviations[p_idx]}: {model.params.X:.3f} (p-value: {model.pvalues.X:.3f})")


def test_bonus_2(x_first=0, x_second=1):
    # "As tests 3/4/5, but using a two-sample Welch's t-test instead of Student's t-test.
    # Compares X=x_first and X=x_second."
    correct_answers = np.array(count_categories())[1]
    incorrect_answers = np.array(count_categories())[2]

    for p_idx in range(correct_answers.shape[0]):
        y_first = np.concatenate((np.ones(correct_answers[p_idx, x_first]), np.zeros(
            incorrect_answers[p_idx, x_first])))
        y_second = np.concatenate((np.ones(correct_answers[p_idx, x_second]), np.zeros(
            incorrect_answers[p_idx, x_second])))
        ttest_result = stats.ttest_ind(y_first, y_second, equal_var=False)

        # Print the results of the two-sample Student's t-test for each prompt
        print(
            f"Two-sample Welch's t-test comparing X={x_first} and X={x_second} for prompt {p_idx+1}: statistic={ttest_result.statistic:.2f}, p-value={ttest_result.pvalue:.4f}")

def print_category_counts():
    print("Raw counts of Correct/Incorrect/Misformatted answers")
    [all_answers, correct_answers, incorrect_answers, misformatted_answers]=count_categories()
    for prompt_number in range(10):
        to_print=[]
        for X_value in range(11):
            to_print.append(f"{correct_answers[prompt_number][X_value]}/{incorrect_answers[prompt_number][X_value]}/{misformatted_answers[prompt_number][X_value]}")
        print(", ".join(to_print))

def print_correct_category_percent():
    print("Percent correct answers:")
    [all_answers, correct_answers, incorrect_answers, misformatted_answers]=count_categories()
    for prompt_number in range(10):
        to_print=[]
        for X_value in range(11):
            to_print.append(f"{correct_answers[prompt_number][X_value]/(incorrect_answers[prompt_number][X_value]+correct_answers[prompt_number][X_value]):.1%}")
        print(", ".join(to_print))

def print_y_percent():
    print("Y(X, P) as percents:")
    [all_answers, correct_answers, incorrect_answers, misformatted_answers]=count_categories()
    for prompt_number in range(10):
        to_print=[]
        for X_value in range(11):
            to_print.append(f"{incorrect_answers[prompt_number][X_value]/(incorrect_answers[prompt_number][X_value]+correct_answers[prompt_number][X_value]):.1%}")
        print(", ".join(to_print))

if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    print("\n")
    test_4()
    print("\n")
    test_5()
    test_6()
    test_bonus_1()
    test_bonus_2(x_first=0, x_second=1)
    test_bonus_2(x_first=1, x_second=10)
    test_bonus_2(x_first=0, x_second=10)
    # print_category_counts()
    # print_correct_category_percent()
    # print_y_percent()