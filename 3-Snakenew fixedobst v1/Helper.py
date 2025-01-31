import matplotlib.pyplot as plt

# Enable interactive mode for plotting
plt.ion()


def plot(scores, mean_scores):
    """
    Plots the scores and mean scores during training.

    Args:
        scores (list): List of scores for each game.
        mean_scores (list): List of mean scores up to each game.
    """
    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")

    plt.plot(scores, label="Score per Game", color="blue")
    plt.plot(mean_scores, label="Mean Score", color="orange")
    plt.ylim(ymin=0)

    # Annotate the latest scores
    if scores:
        plt.scatter(len(scores) - 1, scores[-1], color="blue")
        plt.text(
            len(scores) - 1, scores[-1], f"{scores[-1]}", color="blue", fontsize=10
        )
    if mean_scores:
        plt.scatter(len(mean_scores) - 1, mean_scores[-1], color="orange")
        plt.text(
            len(mean_scores) - 1,
            mean_scores[-1],
            f"{mean_scores[-1]:.2f}",
            color="orange",
            fontsize=10,
        )

    plt.legend()
    plt.draw()
    plt.pause(0.1)
