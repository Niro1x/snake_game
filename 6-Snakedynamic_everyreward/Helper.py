import matplotlib.pyplot as plt
from IPython import display

# Enable interactive mode for plotting
plt.ion()

def plot(scores, mean_scores):
    """
    Plots the scores and mean scores during training.

    Args:
        scores (list): List of scores for each game.
        mean_scores (list): List of mean scores up to each game.
    """
    # Clear the previous plot and display the updated plot
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    # Set up the plot title and labels
    plt.title("Training Progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    # Plot scores and mean scores
    plt.plot(scores, label="Score per Game")
    plt.plot(mean_scores, label="Mean Score")
    plt.ylim(ymin=0)

    # Annotate the latest scores and mean scores
    if scores:
        plt.text(len(scores) - 1, scores[-1], f"{scores[-1]}", color="blue", fontsize=10)
    if mean_scores:
        plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]}", color="orange", fontsize=10)

    # Add a legend for clarity
    plt.legend()

    # Display the updated plot
    plt.show(block=False)
    plt.pause(0.1)
