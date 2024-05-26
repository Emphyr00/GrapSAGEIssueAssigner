import pandas as pd

def load_and_inspect_csv(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Print the structure of the DataFrame
    print("DataFrame Structure:")
    # print(df.info())  # Provides information about DataFrame including the data types and number of non-null values

    # Configure pandas to display the full content of the DataFrame rows
    pd.set_option('display.max_columns', None)  # Ensure all columns are shown
    pd.set_option('display.width', None)        # Ensure the display width is not restricted
    pd.set_option('display.max_colwidth', None) # Display the full content of each column

    # Print a random row from the DataFrame
    random_row = df.sample(n=1)  # n=1 specifies that we want one random sample
    print("\nRandom Row from the Dataset:")
    print(random_row)

# Example usage
if __name__ == "__main__":
    file_path = 'dataset_content.csv'  # Update this to the path of your dataset
    load_and_inspect_csv(file_path)