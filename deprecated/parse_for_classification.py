import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    assert 'class' in df.columns, "CSV must contain 'class' column."
    return df

def filter_by_class(df):
    filtered_df = df[df['class'].notnull()]
    return filtered_df

def process_data(input_csv, output_csv):
    df = load_data(input_csv)
    
    filtered_df = filter_by_class(df)
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data saved to {output_csv}")
    
def process(input_csv, output_csv):
    process_data(input_csv, output_csv)
    
if __name__ == "__main__":
    input_csv = 'dataset_small_classed.csv' 
    output_csv = 'dataset_small_only_classed.csv'
    process_data(input_csv, output_csv)