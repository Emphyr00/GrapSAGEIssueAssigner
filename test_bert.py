import pandas as pd
import torch
from transformers import BertTokenizerFast, BertModel
import nltk
from nltk.util import ngrams

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def load_data(filepath):
    df = pd.read_csv(filepath)
    assert 'extracted_content' in df.columns, "CSV must contain 'extracted_content' column."
    return df['extracted_content']

def initialize_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    return tokenizer, model

def tokenize_texts(tokenizer, text):
    return tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

def get_ngrams(tokens, n=2):
    return list(ngrams(tokens, n))

def remove_stopwords(tokens):
    return [token for token in tokens if token.isalpha() and token not in stop_words]

def get_keywords(tokenizer, tokenized_text, model, n=2):
    with torch.no_grad():
        outputs = model(**tokenized_text)
        attention = outputs.attentions[-1].mean(1).squeeze()  # Simplify to one dimension

    tokens = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'].squeeze().tolist())

    filtered_tokens = remove_stopwords(tokens)

    token_ngrams = get_ngrams(filtered_tokens, n)

    ngram_attention = {}
    for ngram in token_ngrams:
        indices = [tokens.index(token) for token in ngram if token in filtered_tokens]
        if indices:
            average_attention = attention[indices].mean().item()
            ngram_attention[ngram] = average_attention

    sorted_ngrams = sorted(ngram_attention.items(), key=lambda x: x[1], reverse=True)

    top_ngrams = [ngram for ngram, score in sorted_ngrams[:5]]

    return top_ngrams

def main():
    filepath = 'dataset_content.csv'
    contents = load_data(filepath)
    tokenizer, model = initialize_model()

    for content in contents:
        tokenized_text = tokenize_texts(tokenizer, content)
        keywords = get_keywords(tokenizer, tokenized_text, model, n=1)  # Change n to desired n-gram length
        print("Keywords:", keywords)

if __name__ == "__main__":
    main()