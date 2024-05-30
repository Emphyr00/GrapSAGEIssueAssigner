import pandas as pd
import fasttext
import fasttext.util
from nltk.corpus import stopwords
import nltk
from collections import Counter
nltk.download('punkt')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

fasttext.util.download_model('en', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.en.300.bin')

def load_data(filepath):
    df = pd.read_csv(filepath)
    assert 'extracted_content' in df.columns, "CSV must contain 'extracted_content' column."
    return df['extracted_content']

def get_keywords_fasttext(text, model, top_n=5):
    words = [word for word in nltk.word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]
    
    word_vectors = {word: model.get_word_vector(word) for word in words}
    word_frequencies = Counter(words)
    
    keywords = word_frequencies.most_common(top_n)
    return [keyword for keyword, _ in keywords]

def main():
    filepath = 'dataset_content.csv'
    contents = load_data(filepath)

    for content in contents:
        keywords = get_keywords_fasttext(content, fasttext_model, top_n=5)
        print("Keywords:", keywords)

if __name__ == "__main__":
    main()