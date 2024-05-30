import pandas as pd
from collections import Counter
import ast
import numpy as np

def load_data(filepath):
    """Loads the CSV file containing keywords and code_lang."""
    df = pd.read_csv(filepath)
    assert 'keywords' in df.columns and 'code_lang' in df.columns, "CSV must contain 'keywords' and 'code_lang' columns."
    return df

def count_code_langs(df):
    """Counts the occurrences of each code_lang, omitting 'unknown'."""
    return Counter(lang for lang in df['code_lang'] if lang != 'unknown')

def count_keywords(df, code_lang_counts, blacklist):
    """Counts the occurrences of non-code-lang keywords, excluding code_langs and blacklist words."""
    all_keywords = []

    # Aggregate all keywords
    for keywords_str in df['keywords']:
        # Convert string representation of list to actual list
        keywords_list = ast.literal_eval(keywords_str)
        all_keywords.extend(keywords_list)

    # Get the set of code_langs to exclude them from keyword counts
    code_langs_set = set(code_lang_counts.keys())

    # Count occurrences of each keyword, excluding 'unknown', code_langs, and blacklist words
    keyword_counts = Counter(keyword for keyword in all_keywords if keyword != 'unknown' and keyword not in code_langs_set and keyword not in blacklist)

    return keyword_counts

def filter_code_langs(code_lang_counts):
    """Filters code_langs based on a dynamically calculated acceptance threshold."""
    counts = np.array(list(code_lang_counts.values()))
    threshold = np.mean(counts)  # Example: using the mean as a threshold

    filtered_code_lang_counts = {lang: count for lang, count in code_lang_counts.items() if count >= threshold}
    return filtered_code_lang_counts

def find_top_classes(input_csv, top_n=10):
    """Finds the top N non-code-lang keywords and counts of filtered code_langs."""
    # Load the dataset
    df = load_data(input_csv)
    
    # Define blacklist of non-meaningful keywords
    blacklist = {'code', 'data', 'example', 'file', 'files', 'github', 'way'}
    
    # Count code_langs
    code_lang_counts = count_code_langs(df)
    
    # Filter code_langs based on the dynamically calculated threshold
    filtered_code_lang_counts = filter_code_langs(code_lang_counts)
    
    # Count keywords excluding code_langs and blacklist
    keyword_counts = count_keywords(df, filtered_code_lang_counts, blacklist)
    
    # Get the top N non-code-lang keywords
    top_keywords = keyword_counts.most_common(top_n)
    
    # Combine top keywords and filtered code_langs
    relevant_classes = set(keyword for keyword, _ in top_keywords) | set(filtered_code_lang_counts.keys())
    
    print(relevant_classes)
    
    return df, relevant_classes

def add_class_column(df, relevant_classes):
    """Adds a 'class' column to the DataFrame, assigning a class if present in keywords."""
    def find_class(keywords):
        keywords_list = ast.literal_eval(keywords)
        for keyword in keywords_list:
            if keyword in relevant_classes:
                return keyword
        return None

    df['class'] = df['keywords'].apply(find_class)
    return df

def process_data(input_csv, output_csv, top_n=10):
    """Processes the data, finds top classes, and adds a 'class' column."""
    # Find top classes
    df, relevant_classes = find_top_classes(input_csv, top_n)
    
    # Add 'class' column to the DataFrame
    df = add_class_column(df, relevant_classes)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

def process(input_csv, output_csv, top_n=10):
    process_data(input_csv, output_csv, top_n)

if __name__ == "__main__":
    input_csv = 'dataset_small_extracted.csv'
    output_csv = 'dataset_small_classed.csv'
    process_data(input_csv, output_csv, top_n=10)