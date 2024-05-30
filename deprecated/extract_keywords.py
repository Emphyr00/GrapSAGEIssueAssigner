import pandas as pd
import fasttext
import fasttext.util
import spacy
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from guesslang import Guess

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))

# Load FastText model
fasttext.util.download_model('en', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.en.300.bin')

# Initialize Guesslang
guess = Guess()

def load_data(filepath, fraction=1.0):
    """Loads a fraction of the CSV file containing extracted content and code."""
    df = pd.read_csv(filepath)
    assert 'extracted_content' in df.columns and 'code' in df.columns, "CSV must contain 'extracted_content' and 'code' columns."
    df = df.sample(frac=fraction, random_state=1).reset_index(drop=True)  # Take a sample of the dataframe
    return df

def get_keywords_fasttext(text, model, top_n=5):
    """Extracts keywords that are nouns from text using FastText model."""
    if pd.isna(text):
        return []
    
    # Use spaCy to process the text
    doc = nlp(text.lower())
    
    # Extract nouns
    words = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN') and token.is_alpha and token.text not in stop_words]
    
    # Get word vectors
    word_vectors = {word: model.get_word_vector(word) for word in words}
    
    if not word_vectors:
        return []
    
    # Compute the centroid of the word vectors
    centroid = np.mean(list(word_vectors.values()), axis=0)
    
    # Calculate cosine similarity between each word vector and the centroid
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    similarities = {word: cosine_similarity(vector, centroid) for word, vector in word_vectors.items()}
    
    # Sort words by similarity and get the top N keywords
    sorted_words = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    keywords = [word for word, _ in sorted_words[:top_n]]
    
    print(keywords)
    
    return keywords

def detect_language(code):
    """Detects the programming language of a code snippet using guesslang."""
    if not code or pd.isna(code):
        return 'unknown'
    
    lang = guess.language_name(code)
    # Every broken snipnet is marked as ini so better to get rid of it
    if lang == 'INI':
        return 'unknown'
    
    print(lang)
    
    if lang == None:
        return 'unknown'
    
    return lang.lower()

def process_data(input_csv, output_csv, fraction=0.1):
    """Processes a fraction of the input CSV, extracts keywords and programming language, and saves the results to an output CSV."""
    # Load the CSV file
    df = load_data(input_csv, fraction)
    
    # Extract keywords and programming language
    df['code_lang'] = df['code'].apply(detect_language)
    df['keywords'] = df.apply(lambda row: get_keywords_fasttext(row['extracted_content'], fasttext_model, top_n=5) + ([row['code_lang']] if row['code_lang'] != 'unknown' else []), axis=1)
    
    # Save the results to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")
    
def process(input_csv, output_csv):
    process_data(input_csv, output_csv)

if __name__ == "__main__":
    input_csv = 'dataset_small_content.csv'  
    output_csv = 'dataset_small_extracted.csv'
    process_data(input_csv, output_csv)