import pandas as pd
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_dataset(filepath):
    return pd.read_csv(filepath)

def append_to_csv(df, filepath):
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        df.to_csv(f, header=f.tell() == 0, index=False)

def load_existing_records(filepath):
    if not os.path.exists(filepath):
        return None, set()

    df = pd.read_csv(filepath)
    if df.empty:
        return None, set()

    processed_records = set(df['record'])
    last_record = df['record'].iloc[-1]
    return last_record, processed_records

def main():
    input_filepath = 'dataset_content.csv'
    output_filepath = 'dataset_tagged.csv'

    df = load_dataset(input_filepath)

    last_record, processed_records = load_existing_records(output_filepath)

    if last_record:
        start_index = df[df['extracted_content'] == last_record].index[0] + 1
    else:
        start_index = 0

    for index in range(start_index, len(df)):
        clear_console()
        print("Record #{}".format(index + 1))
        row = df.iloc[index]
        print(row['extracted_content'])

        if row['extracted_content'] in processed_records:
            continue

        user_decision = input("Keep this record? (y/n): ")
        if user_decision.lower() == 'y':
            keywords = input("Enter tags for this record, separated by commas: ")
            tagged_data = pd.DataFrame({'record': [row['extracted_content']], 'tags': [keywords]})
            append_to_csv(tagged_data, output_filepath)
            print("Record saved.")
        else:
            print("Record skipped.")

    print("Tagging complete. Data saved to '{}'.".format(output_filepath))

if __name__ == "__main__":
    main()