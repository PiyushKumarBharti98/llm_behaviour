import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
INPUT_CSV_FILE = "llm_persona_analysis_multi_question_results.csv"
AGGREGATE_PLOT_FILE = "aggregate_results.png"
PER_TOPIC_PLOT_FILE = "per_topic_consistency.png"


def create_visualizations():
    """
    Loads the experimental data and generates the aggregate and per-topic
    visualizations for the research findings.
    """
    # Check if the input file exists
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Error: The data file '{INPUT_CSV_FILE}' was not found.")
        print("Please make sure you have run the main experiment script first.")
        return

    print(f"Loading data from '{INPUT_CSV_FILE}'...")
    df = pd.read_csv(INPUT_CSV_FILE)

    # Set a professional plot style
    sns.set_theme(style="whitegrid")

    # --- Graph 1: Aggregate Results Summary ---
    print(f"Generating aggregate results plot -> '{AGGREGATE_PLOT_FILE}'")

    # Melt the dataframe to make it suitable for seaborn's barplot
    df_agg = (
        df.groupby("track")[
            ["hedging_word_count", "nli_entailment_score", "sentiment_compound"]
        ]
        .mean()
        .reset_index()
    )
    df_melted_agg = df_agg.melt(
        id_vars="track", var_name="metric", value_name="average_score"
    )

    # Create a mapping for more readable metric names
    metric_labels = {
        "hedging_word_count": "Hedging Count",
        "nli_entailment_score": "Directness Score (NLI)",
        "sentiment_compound": "Sentiment Score",
    }
    df_melted_agg["metric"] = df_melted_agg["metric"].map(metric_labels)

    plt.figure(figsize=(12, 7))
    barplot_agg = sns.barplot(
        data=df_melted_agg,
        x="metric",
        y="average_score",
        hue="track",
        palette={"Cooperative": "cornflowerblue", "Confrontational": "salmon"},
    )
    plt.title(
        "Aggregate Comparison of LLM Persona Metrics (All Topics)",
        fontsize=16,
        weight="bold",
    )
    plt.ylabel("Average Score", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.xticks(rotation=0, ha="center", fontsize=11)
    plt.legend(title="Track", fontsize=11)

    # Add data labels on top of the bars
    for p in barplot_agg.patches:
        barplot_agg.annotate(
            format(p.get_height(), ".2f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(AGGREGATE_PLOT_FILE)
    plt.close()

    # --- Graph 2: Per-Topic Consistency ---
    print(f"Generating per-topic consistency plot -> '{PER_TOPIC_PLOT_FILE}'")

    # We use catplot to create a faceted grid, showing the results for each topic
    g = sns.catplot(
        data=df,
        x="track",
        y="hedging_word_count",  # Focus on the most significant metric
        col="question_topic",
        kind="bar",
        col_wrap=3,
        height=4,
        aspect=1,
        palette={"Cooperative": "cornflowerblue", "Confrontational": "salmon"},
        legend=False,
    )
    g.fig.suptitle(
        "Per-Topic Consistency: Hedging Behavior Increases Across All Subjects",
        fontsize=16,
        weight="bold",
        y=1.03,
    )
    g.set_axis_labels("Track", "Average Hedging Word Count")
    g.set_titles("Topic: {col_name}")
    g.tight_layout(rect=[0, 0, 1, 0.97])

    # Add data labels to each bar in the catplot
    for ax in g.axes.flatten():
        for p in ax.patches:
            ax.annotate(
                format(p.get_height(), ".1f"),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=9,
            )

    plt.savefig(PER_TOPIC_PLOT_FILE)
    plt.close()

    print("\nVisualizations have been saved successfully!")


if __name__ == "__main__":
    create_visualizations()
