import os

# Root directory where the "id" folders are located
root_directory = "/Users/a107191/Documents/test_directory"

# Iterate through the "id" folders
for id_folder in os.listdir(root_directory):

    if (not id_folder.startswith(".")):

        id_folder_path = os.path.join(root_directory, id_folder)
        print(id_folder_path)

        # Check if it's a directory
        if os.path.isdir(id_folder_path):
            # Subfolder with arbitrary name (assuming only one subfolder)
            subfolder = os.listdir(id_folder_path)[0]
            if (not subfolder.startswith(".")):
                subfolder_path = os.path.join(id_folder_path, subfolder)
            else:
                subfolder_path = os.path.join(id_folder_path, os.listdir(id_folder_path)[1])
            # print(subfolder_path)

            # Initialize an empty string to store merged content
            merged_content = ""

            # Iterate through the text files and merge their contents
            for i in range(1, 6):
                filename = f"{id_folder}-Q-{i}.webm.txt"
                # print(filename)
                file_path = os.path.join(subfolder_path, filename)
                # print(file_path)

                # Check if the file exists
                if os.path.isfile(file_path):
                    with open(file_path, "r") as file:
                        content = file.read()
                        merged_content += content

            # Create a merged file and write the combined content
            merged_file_path = os.path.join(root_directory, id_folder+".txt")
            with open(merged_file_path, "w") as merged_file:
                merged_file.write(merged_content)

            print(f"Merged file for {id_folder}: {merged_file_path}")