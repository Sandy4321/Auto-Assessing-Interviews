import os
import pandas as pd


root_directory = "/Users/a107191/Downloads/transcripts_whisper/"


csv_file_path = "/Users/a107191/Downloads/transcripts_whisper/whisper_transcripts_dataset.csv"
df = pd.read_csv(csv_file_path)

new_columns = ["whisper_ts1", "whisper_ts2", "whisper_ts3", "whisper_ts4", "whisper_ts5"]

for column in new_columns:
    df[column] = ""


for index, row in df.iterrows():
    id_value = row['videoDate_x']
    id_folder_path = os.path.join(root_directory, str(id_value))


    if os.path.exists(id_folder_path):
        # Iterate through the range 1 to 5 to extract the contents of text files

        if os.path.isdir(id_folder_path):
        # Subfolder with arbitrary name (assuming only one subfolder)
            subfolder = os.listdir(id_folder_path)[0]
            if (not subfolder.startswith(".")):
                subfolder_path = os.path.join(id_folder_path, subfolder)
            else:
                subfolder_path = os.path.join(id_folder_path, os.listdir(id_folder_path)[1])

        for i in range(1, 6):
            filename = f"{id_value}-Q-{i}.webm.txt"
            file_path = os.path.join(subfolder_path, filename)
            # print(file_path)
            # Check if the file exists
            if os.path.isfile(file_path):
                with open(file_path, "r") as file:
                    content = file.read()
                    # Update the corresponding new column
                    df.at[index, f"whisper_ts{i}"] = content

# Save the updated DataFrame back to the CSV file
df.to_csv(csv_file_path, index=False)