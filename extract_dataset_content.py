import pandas as pd
import re

def clean_and_extract_content(text):
    """Removes specific tags and extracts content after the first <issue_comment> tag"""
    # Remove <issue_closed> tags
    cleaned_text = re.sub(r'<issue_closed>', '', text)
    # Remove username_0: patterns
    cleaned_text = re.sub(r'username_\d+:', '', cleaned_text)
    # Remove @username_0 patterns
    cleaned_text = re.sub(r'@username_\d', '', cleaned_text)
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
        return content

    return ""  # Return empty if no <issue_comment> tag is found

def process_csv(input_csv, output_csv):
    """Processes the input CSV and saves the cleaned content to an output CSV"""
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # Assume 'content' is the name of the column containing the issue text
    df['extracted_content'] = df['content'].apply(clean_and_extract_content)

    # Save only the extracted_content column to a new CSV file
    df[['extracted_content']].to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    input_csv = 'dataset.csv'  # Update this to the path of your input CSV
    output_csv = 'dataset_content3.csv'  # Name of the output CSV file
    process_csv(input_csv, output_csv)