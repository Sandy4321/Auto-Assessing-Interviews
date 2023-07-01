from transformers import pipeline
feedback_classifier = pipeline(task = "zero-shot-classification", model = "facebook/bart-large-mnli")
feedback_labels = ["body language & gestures", "conversational skill", "storytelling", "self promotion", "positive impression", "video issue", "sound issue", "confidence", "talk through", "behaviour"]


import pandas as pd
feedback_df = pd.read_csv("Interview_Transcripts_Analysis.csv")

feedback_df.head()

from tqdm import tqdm

for i in tqdm(range(feedback_df.shape[0])):
    interview_feedback = feedback_df.loc[i].at["overallComments"]
    classification_result = feedback_classifier(interview_feedback, feedback_labels, multi_label = True)
    length = len(classification_result["labels"])
    for j in range (length):
        label_name = classification_result["labels"][j]
        label_score = classification_result["scores"][j]
        feedback_df.at[i, label_name] = label_score

feedback_df.head(40)


feedback_df.to_csv("classified_feedback.csv")
