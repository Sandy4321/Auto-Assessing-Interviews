import os
import pandas as pd

directory = "/home/pooja/pooja/unil/transcripts/transcripts/"
test_files = []

print("hello")
df = pd.read_csv("self_promotion_scores_transcripts.csv")
train_files = df["videoDate"].tolist()
train_files = map(str, train_files)
train_files = list(train_files)
print(len(train_files))
print(type(train_files[1]))
print(train_files[1])

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        f = os.path.splitext(f)[0]
        f = f[47:]
        f = f[:-11]

        if f not in train_files:
            test_files.append(f)

test_files_set = set(test_files)
unique_test_files = list(test_files_set)

print(unique_test_files)

