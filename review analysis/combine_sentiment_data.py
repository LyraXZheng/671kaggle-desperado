import os
import json
import re

# Define the folder paths in the specified order
folders = ["test_part1", "test_part2"]
base_path = r"../../data/review data/sentiment_output"
output_data = []

# Loop through each folder in the specified order
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.json')],
        key=lambda x: (int(re.search(r'thread_(\d+)_part_(\d+)', x).group(1)),
                       int(re.search(r'thread_(\d+)_part_(\d+)', x).group(2)))
    )

    # Load each JSON file in the correct order and add it to the output_data list
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            output_data.append(data)

# Save the merged data to a new file
output_file = r"../../data/review data/review_sentiment_test.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("Merging complete! File saved as", output_file)
