import pandas as pd
from evaluate import load
bertscore = load("bertscore")

predictions_df = pd.read_csv("predictions.csv")

predictions_list = predictions_df["Generated Text"].tolist()
references_list = predictions_df["Actual Text"].tolist()

results = bertscore.compute(predictions=predictions_list, references=references_list, lang="en")
print(results)
