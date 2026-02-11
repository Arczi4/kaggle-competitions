import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def compare_hists(frame, cols):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(cols),
        figsize=(25, 4),
        sharey=False
    )

    for ax, col in zip(axes, cols):
        sns.boxplot(
            data=frame,
            x="loan_paid_back",
            y=col,
            hue="loan_paid_back",
            palette=["m", "g"],
            legend=False,
            showfliers=False,
            ax=ax
        )
    ax.set_title(col)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.show()


def compare_counts(frame, cols):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(cols)//2,
        figsize=(25, 6),
        sharey=False
    )

    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        sns.countplot(
            data=frame,
            x=col,
            hue="loan_paid_back",
            ax=ax
        )
        ax.set_title(col)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.show()


def missing_data(df: pd.DataFrame, n: int):
    """
    Displays DataFrame with info about missing data per feature
    
    :param df: df used for analysing of nulls
    :type df: pd.DataFrame
    :param n: number of null rows to display
    :type n: int
    """
    null_sum = df.isnull().sum()
    total = null_sum.sort_values(ascending=False) # Total number of missing values
    percentage = (null_sum / df.isnull().count()).sort_values(ascending=False) * 100  # % of Missing values
    unique_values = df.nunique()
    missing_data = pd.concat([total, percentage, unique_values], axis=1, 
                             keys=['Missing', '% Missing', 'Unique values'], sort=False)
    
    return missing_data.head(n)