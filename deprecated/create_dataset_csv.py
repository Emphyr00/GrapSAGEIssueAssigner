import pandas as pd
import os
import csv

def read_and_merge_parquet(directory_path, output_csv_file):
    data_frames = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(directory_path, file_name)
            df = pd.read_parquet(file_path, engine='pyarrow')
            data_frames.append(df)
    
    merged_df = pd.concat(data_frames, ignore_index=True)
    
    # Save the merged DataFrame to CSV, specifying escape character and quoting
    merged_df.to_csv(output_csv_file, index=False, escapechar='\\', quoting=csv.QUOTE_ALL)

    print(f"Merged data saved to {output_csv_file}")

# Example usage
if __name__ == "__main__":
    directory_path = 'dataset'  # Update this to the path of your directory
    output_csv_file = 'dataset_small.csv'  # Name of the output CSV file
    read_and_merge_parquet(directory_path, output_csv_file)