import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. LOAD THE DATA
# ==========================================
print("Loading evaluation results...")
try:
    df = pd.read_csv("final_evaluation_results.csv")
except FileNotFoundError:
    print("Error: 'final_evaluation_results.csv' not found.")
    exit(1)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# ==========================================
# GRAPH 1: HEAD-TO-HEAD SCATTER PLOT
# ==========================================
print("Generating Scatter Plot...")
plt.figure(figsize=(7, 7))

# Plot the dots
sns.scatterplot(x="RAG_F1", y="QLoRA_F1", data=df, s=80, color="#4C72B0", alpha=0.7)

# Draw the diagonal "Tie" line (y = x)
limits = [max(df['RAG_F1'].min(), df['QLoRA_F1'].min()) - 0.05, 1.05]
plt.plot(limits, limits, linestyle='--', color='red', label="Tie Line (RAG = QLoRA)")

plt.title("Query-by-Query Performance: RAG vs QLoRA", fontweight='bold', pad=15)
plt.xlabel("RAG BERTScore F1", fontsize=12)
plt.ylabel("QLoRA BERTScore F1", fontsize=12)
plt.xlim(limits)
plt.ylim(limits)

# Add annotations to explain the zones
plt.text(limits[0] + 0.02, limits[1] - 0.05, "QLoRA Performed Better ->", color='gray', fontsize=10)
plt.text(limits[1] - 0.25, limits[0] + 0.02, "<- RAG Performed Better", color='gray', fontsize=10)

plt.legend()
plt.tight_layout()
plt.savefig("thesis_fig_scatterplot.pdf", dpi=300, format="pdf")
plt.close()

# ==========================================
# GRAPH 2: WIN/TIE/LOSS STACKED BAR CHART
# ==========================================
print("Generating Win/Tie/Loss Chart...")
plt.figure(figsize=(8, 3))

# Define what counts as a "Tie" (e.g., scores within 0.02 of each other)
tie_margin = 0.02

# Categorize each row
conditions = [
    (df['RAG_F1'] > df['QLoRA_F1'] + tie_margin),
    (df['QLoRA_F1'] > df['RAG_F1'] + tie_margin)
]
choices = ['RAG Win', 'QLoRA Win']
df['Result'] = np.select(conditions, choices, default='Tie')

# Count the occurrences
results_count = df['Result'].value_counts(normalize=True) * 100 # Get percentages

# Prepare data for plotting
plot_df = pd.DataFrame(results_count).T
# Ensure all columns exist even if count is 0
for col in ['RAG Win', 'Tie', 'QLoRA Win']:
    if col not in plot_df.columns:
        plot_df[col] = 0

# Colors for RAG Win, Tie, QLoRA Win
colors = ["#55A868", "#C44E52", "#DD8452"]

plot_df[['RAG Win', 'Tie', 'QLoRA Win']].plot(
    kind='barh', stacked=True, color=colors, figsize=(8, 2.5), width=0.4
)

plt.title("Overall Win Rate (% of Queries)", fontweight='bold', pad=15)
plt.xlabel("Percentage (%)", fontsize=12)
plt.yticks([]) # Hide Y-axis labels
plt.xlim(0, 100)

# Move legend outside
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

plt.tight_layout()
plt.savefig("thesis_fig_winrate.pdf", dpi=300, format="pdf", bbox_inches="tight")
plt.close()

# ==========================================
# GRAPH 3: OVERLAPPING DENSITY PLOT (KDE)
# ==========================================
print("Generating Density Plot...")
plt.figure(figsize=(8, 5))

# Plot smooth distributions
sns.kdeplot(data=df, x="RAG_F1", fill=True, label="RAG Architecture", color="#4C72B0", alpha=0.5)
sns.kdeplot(data=df, x="QLoRA_F1", fill=True, label="QLoRA Fine-Tune", color="#DD8452", alpha=0.5)

plt.title("Density Distribution of Evaluation Scores", fontweight='bold', pad=15)
plt.xlabel("BERTScore F1", fontsize=12)
plt.ylabel("Density (Frequency)", fontsize=12)
plt.xlim(0, 1.05)
plt.legend(loc="upper left")

plt.tight_layout()
plt.savefig("thesis_fig_density.pdf", dpi=300, format="pdf")
plt.close()

print("\nSuccess! Generated 3 new advanced visualizations:")
print("- thesis_fig_scatterplot.pdf")
print("- thesis_fig_winrate.pdf")
print("- thesis_fig_density.pdf")