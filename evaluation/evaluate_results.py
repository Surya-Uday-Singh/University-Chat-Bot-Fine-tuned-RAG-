import pandas as pd
import evaluate

# ==========================================
# 1. LOAD AND MERGE THE DATA
# ==========================================
print("Loading generated outputs...")
try:
    df_rag = pd.read_csv("rag_outputs.csv")
    df_qlora = pd.read_csv("qlora_outputs.csv")
except FileNotFoundError as e:
    print(f"Error: Could not find the CSV files. Make sure they are in the same folder. Details: {e}")
    exit(1)

print("Merging datasets side-by-side...")
# The 'inner' join ensures we only evaluate questions that exist in both files
df_combined = pd.merge(df_rag, df_qlora, on="Instruction", how="inner")

# ==========================================
# 2. INITIALIZE THE METRIC
# ==========================================
print("Loading BERTScore metric (this may take a moment to download the model weights)...")
bertscore = evaluate.load("bertscore")

# ==========================================
# 3. CALCULATE SCORES
# ==========================================
print("Calculating BERTScores for RAG...")
rag_results = bertscore.compute(
    predictions=df_combined["RAG_Answer"].tolist(),
    references=df_combined["Golden_Answer"].tolist(),
    lang="en"
)
# Save the F1 scores as a new column in our dataframe
df_combined["RAG_F1"] = rag_results["f1"]

print("Calculating BERTScores for QLoRA...")
qlora_results = bertscore.compute(
    predictions=df_combined["QLoRA_Answer"].tolist(),
    references=df_combined["Golden_Answer"].tolist(),
    lang="en"
)
# Save the F1 scores as a new column
df_combined["QLoRA_F1"] = qlora_results["f1"]

# ==========================================
# 4. SAVE AND SUMMARIZE
# ==========================================
output_filename = "final_evaluation_results.csv"
df_combined.to_csv(output_filename, index=False)

print("\n" + "="*45)
print("🎓 FINAL EVALUATION SUMMARY")
print("="*45)
print(f"Total Questions Evaluated: {len(df_combined)}")
print(f"Average RAG F1 Score:   {df_combined['RAG_F1'].mean():.4f}")
print(f"Average QLoRA F1 Score: {df_combined['QLoRA_F1'].mean():.4f}")
print("="*45)
print(f"\nDetailed row-by-row results saved to: {output_filename}")