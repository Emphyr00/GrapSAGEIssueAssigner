import pandas as pd
import re

def clean_and_extract_content(text):
    # Remove <issue_closed> tags
    cleaned_text = re.sub(r'<issue_closed>', '', text)
    # Remove username_0: patterns
    cleaned_text = re.sub(r'username_\d+:', '', cleaned_text)
    # Remove @username_\d patterns
    cleaned_text = re.sub(r'@username_\d', '', cleaned_text)
     # Remove image tags and similar patterns
    cleaned_text = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned_text)
    
    # Extract code blocks (enclosed in triple or single backticks)
    code_blocks = re.findall(r'`{1,3}(.*?)`{1,3}', cleaned_text, re.S)
    code = ' '.join(code_blocks).strip()
    
    # Remove code blocks from the text
    cleaned_text = re.sub(r'`{1,3}.*?`{1,3}', '', cleaned_text, flags=re.S)
    
    # Extract content after the first <issue_comment> tag
    matches = re.search(r'<issue_comment>(.*?)(?:<issue_comment>|$)', cleaned_text, re.S)
    if matches:
        content = matches.group(1).strip()
        # Remove the initial 'Title:' if it is present in the extracted content
        content = re.sub(r'^Title:\s*', '', content)
        # Replace newline and carriage return characters
        content = content.replace('\n', ' ').replace('\r', ' ')
        # Normalize multiple spaces to a single space
        content = re.sub(r'\s+', ' ', content)
        return content, code

    return "", code  # Return empty content and code if no <issue_comment> tag is found

def process_csv(input_csv, output_csv):
    """Processes the input CSV and saves the cleaned content and code to an output CSV"""
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Assume 'content' is the name of the column containing the issue text
    df[['extracted_content', 'code']] = df['content'].apply(lambda text: pd.Series(clean_and_extract_content(text)))

    # Save the extracted content and code columns to a new CSV file
    df[['extracted_content', 'code']].to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

def process(input_csv, output_csv):
    process_csv(input_csv, output_csv)

if __name__ == "__main__":
    input_csv = 'dataset_small.csv'  # Update this to the path of your input CSV
    output_csv = 'dataset_small_content.csv'  # Name of the output CSV file
    process_csv(input_csv, output_csv)
