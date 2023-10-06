import pandas as pd

df = pd.read_csv("/Users/a107191/Downloads/Merged_Final.csv")

print(df.head())

print(df['videoDate_x'].nunique())

l = df.videoDate_x.unique().tolist()

file_names_to_pick = [str(f)+".txt" for f in l]

import os
import shutil

source_folder = "/Users/a107191/Documents/merged_whisper_transcripts/"
destination_folder = "/Users/a107191/Downloads/whisper_transcripts_with_feedback/"

for filename in os.listdir(source_folder):
    if filename in file_names_to_pick:
        source_file_path = os.path.join(source_folder, filename)
        destination_file_path = os.path.join(destination_folder, filename)
        shutil.copy(source_file_path, destination_file_path)