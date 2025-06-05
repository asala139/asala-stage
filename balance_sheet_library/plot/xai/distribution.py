import matplotlib.pyplot as plt

def top_and_worst_distribution(top_unique, top_count, last_unique, last_count, save=False, show=True,
                               filename='top_wors_comparison.png'):
    fig, ax = plt.subplots(2)
    fig.set_figwidth(9)
    fig.suptitle("Distribution of top 50 and worst 50 features among finance statements")
    fig.subplots_adjust(hspace=0.5)
    ax[0].set_title("Top 50")
    ax[0].bar(top_unique, top_count, color='seagreen', alpha=0.5)
    ax[0].set_xticks([7, 94, 217, 276])
    ax[0].set_xticklabels(["Financial Profile", "Balance Sheet", "Income Statement", "Ratio Analysis"])
    ax[1].set_title("Worst 50")
    ax[1].bar(last_unique, last_count, color='tomato', alpha=0.5)
    ax[1].set_xticks([7, 94, 217, 276])
    ax[1].set_xticklabels(["Financial Profile", "Balance Sheet", "Income Statement", "Ratio Analysis"])
    if save:
        fig.savefig(filename)
    if show:
        fig.show()