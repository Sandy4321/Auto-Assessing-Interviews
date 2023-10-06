import pandas as pd
import os


df = pd.read_csv('/Users/a107191/Downloads/whisper_transcripts_with_feedback/Merged_Final.csv')

text_files_directory = '/Users/a107191/Downloads/whisper_transcripts_with_feedback/'

def read_text_file(row):
    file_path = os.path.join(text_files_directory, f"{row['videoDate_x']}.txt")
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return None


df['whisper_transcripts'] = df.apply(read_text_file, axis=1)

df['whisper_transcripts'] = df['whisper_transcripts'].str.replace('\n', '', regex=True)

df = df[df["whisper_transcripts"].apply(lambda x: x != 'None' if isinstance(x, str) else not pd.isna(x))]

df.to_csv("whisper_transcripts_dataset.csv", index=False)

print(df.shape)

print(df.head())