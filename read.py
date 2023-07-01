# import pandas as pd
# reddit_df = pd.read_csv("reddit_filtered.csv")

# reddit_df.drop(reddit_df.tail(52000).index,inplace = True)

# print(reddit_df.head())

import pandas as pd
overall_comments_df = pd.read_excel("overall_comments.xlsx", sheet_name=0, header=0, index_col=False, keep_default_na=True)

print(overall_comments_df.shape)

print(overall_comments_df.to_string())
