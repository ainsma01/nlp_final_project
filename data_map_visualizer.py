import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_datamap_from_dir(directory):
    """
    Reads all JSONL files in a directory and concatenates them into a single DataFrame.
    """
    records = []
    for fname in os.listdir(directory):
        if fname.endswith(".jsonl"):
            path = os.path.join(directory, fname)
            with open(path, "r") as f:
                for line in f:
                    records.append(json.loads(line))
    df = pd.DataFrame(records)
    return df

def categorize(row):
    """
    Categorize each feature/example into easy, ambiguous, or hard.
    """
    if row['confidence'] > 0.8 and row['variability'] < 0.5:
        return "easy"
    elif row['confidence'] < 0.5 and row['variability'] > 1.0:
        return "hard"
    else:
        return "ambiguous"

def plot_datamap(df, save_path=None):
    """
    Plots the dataset cartography scatter plot.
    """
    df['category'] = df.apply(categorize, axis=1)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="variability",
        y="confidence",
        hue="category",
        palette={"easy":"green", "hard":"red", "ambiguous":"orange"},
        alpha=0.6
    )
    plt.xlabel("Variability (std of loss)")
    plt.ylabel("Confidence (mean P(correct))")
    plt.title("Dataset Cartography")
    plt.legend(title="Category")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    
    plt.show()

# ---- Usage ----

data_dir = "data_maps/"  # directory containing all data_map_epoch_X.jsonl files
df = load_datamap_from_dir(data_dir)

# Optional: aggregate multiple features per example_id
df_example = df.groupby("example_id").agg({
    "correctness": "mean",
    "variability": "mean",
    "confidence": "mean"
}).reset_index()

# Plot feature-level cartography
plot_datamap(df, save_path="feature_cartography.png")

# Plot example-level cartography
plot_datamap(df_example, save_path="example_cartography.png")
