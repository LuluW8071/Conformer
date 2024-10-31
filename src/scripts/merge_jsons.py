import os
import json
import random
import argparse

def load_and_merge_json_files(file_paths):
    merged_data = []
    for file_path in file_paths:
        if os.path.isfile(file_path) and file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                merged_data.extend(data)
    return merged_data

def shuffle_data(data):
    random.shuffle(data)
    return data

def save_to_json(data, output_file):
    with open(output_file, 'w') as out_file:
        json.dump(data, out_file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Merge and shuffle JSON files.')
    parser.add_argument('files', metavar='F', type=str, nargs='+', 
                        help='JSON files to merge')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output file for merged JSON')

    args = parser.parse_args()

    merged_data = load_and_merge_json_files(args.files)
    shuffled_data = shuffle_data(merged_data)
    save_to_json(shuffled_data, args.output)

    print(f"Merging and shuffling complete. Output saved to '{args.output}'.")

if __name__ == "__main__":
    main()
