import pandas as pd

self_promotion = pd.read_csv("self_promotion_scores_transcripts.csv")

#print("Shape of dataframe is: ", self_promotion.shape)

#print(self_promotion.head())

#print(self_promotion.columns)

#print(self_promotion["interview"].head(30))

#print(self_promotion.to_string())

#print(self_promotion['interview'].isnull().sum()

storytelling_examples = self_promotion.loc[self_promotion['overallComments'].str.contains("storytelling", case=False)]

#print(storytelling_examples['overallComments'].to_string(index=False))

print(storytelling_examples.shape)
