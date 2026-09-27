import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD AND PREPARE THE DATA
# ==========================================
print("Loading evaluation results...")
try:
    df = pd.read_csv("final_evaluation_results.csv")
except FileNotFoundError:
    print("Error: 'final_evaluation_results.csv' not found.")
    exit(1)

# To use Seaborn effectively, we need to "melt" the dataframe.
# This reshapes the data from wide format (columns for RAG_F1 and QLoRA_F1)
# to long format (one column for 'Model', one column for 'Score').
df_melted = df.melt(
    id_vars=["Instruction"],
    value_vars=["RAG_F1", "QLoRA_F1"],
    var_name="Model",
    value_name="BERTScore (F1)"
)

# Clean up the labels so they look nice on the graph (e.g., 'RAG_F1' -> 'RAG')
df_melted["Model"] = df_melted["Model"].str.replace("_F1", "")

# Set the overall academic aesthetic
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# ==========================================
# 2. GENERATE GRAPH A: THE BOX PLOT
# ==========================================
print("Generating Box Plot...")
plt.figure(figsize=(8, 6))

# The boxplot shows the median, quartiles, and any wild outliers
sns.boxplot(
    x="Model",
    y="BERTScore (F1)",
    data=df_melted,
    palette="Set2",
    width=0.5,
    hue="Model", # Added hue to avoid future seaborn warnings
    legend=False
)

plt.title("Distribution of Semantic Similarity (BERTScore F1)", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("BERTScore F1", fontsize=12)
plt.xlabel("Architecture", fontsize=12)
# Set Y-axis to 0-1 since BERTScores are percentages
plt.ylim(0, 1.05)

plt.tight_layout()
boxplot_filename = "thesis_fig_boxplot.pdf"
plt.savefig(boxplot_filename, dpi=300, format="pdf") # PDF is best for thesis documents (no pixelation)
plt.close() # Close the figure to free up memory

# ==========================================
# 3. GENERATE GRAPH B: THE BAR CHART
# ==========================================
print("Generating Bar Chart...")
plt.figure(figsize=(7, 5))

# Calculate the mean scores for the bar chart
mean_scores = df[["RAG_F1", "QLoRA_F1"]].mean().reset_index()
mean_scores.columns = ["Model", "Average Score"]
mean_scores["Model"] = mean_scores["Model"].str.replace("_F1", "")

# Plot the bars
ax = sns.barplot(
    x="Model",
    y="Average Score",
    data=mean_scores,
    palette="mako",
    hue="Model",
    legend=False
)

plt.title("Average Semantic Similarity: RAG vs. QLoRA", fontsize=14, fontweight='bold', pad=15)
plt.ylabel("Average BERTScore F1", fontsize=12)
plt.xlabel("Architecture", fontsize=12)
plt.ylim(0, 1.05)

# Add the exact numerical values on top of each bar
for index, row in mean_scores.iterrows():
    plt.text(
        index,
        row["Average Score"] + 0.02,
        f'{row["Average Score"]:.3f}',
        color='black',
        ha="center",
        fontweight='bold'
    )

plt.tight_layout()
barchart_filename = "thesis_fig_barchart.pdf"
plt.savefig(barchart_filename, dpi=300, format="pdf")
plt.close()

print("\nSuccess! Your graphs have been saved as:")
print(f"- {boxplot_filename}")
print(f"- {barchart_filename}")