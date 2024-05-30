import os
import pandas as pd
import re
import fasttext
import fasttext.util
import spacy
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from whats_that_code.election import guess_language_all_methods

nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
fasttext.util.download_model('en', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.en.300.bin')

class DatasetParser:
    def __init__(self, directory_path, output_csv):
        self.directory_path = directory_path
        self.output_csv = output_csv
        self.df = None
           
    def create_csv(self, output_file=None):
        self.df.to_csv(output_file or self.output_csv, index=False)
        return self
    
    def get_df(self):
        return self.df
        
    def read_and_merge_parquet(self, size=1.0):
        data_frames = []
        for file_name in os.listdir(self.directory_path):
            if file_name.endswith('.parquet'):
                file_path = os.path.join(self.directory_path, file_name)
                df = pd.read_parquet(file_path, engine='pyarrow')
                
                df = df[['repo', 'content']]
                
                # Sample the specified fraction of the dataframe
                if 0 < size < 1:
                    df = df.sample(frac=size)
                
                data_frames.append(df)
        
        self.df = pd.concat(data_frames, ignore_index=True)
        return self
        
    def clean_and_extract_content(self, text):
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

        return "", code
    
    def parse_content(self):
        self.df[['extracted_content', 'code']] = self.df['content'].apply(lambda text: pd.Series(self.clean_and_extract_content(text)))
        
        self.df = self.df[['repo', 'extracted_content', 'code']]
        return self
    
    def get_keywords_fasttext(self, text, model, top_n=5):
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
        
        return keywords

    def detect_language(self, code):
        """Detects the programming language of a code snippet using guesslang."""
        if not code or pd.isna(code):
            return 'unknown'

        # shity library that breaks sometime without any reason
        try:
            lang = guess_language_all_methods(code)
        except:
            lang = 'unknown'
        if lang == 'text only':
            return 'unknown'
        
        if lang == None:
            return 'unknown'
        
        return lang.lower()
    
    def extract_keywords_and_code(self):
        self.df['code_lang'] = self.df['code'].apply(self.detect_language)
        self.df['keywords'] = self.df.apply(lambda row: self.get_keywords_fasttext(row['extracted_content'], fasttext_model, top_n=5) + ([row['code_lang']] if row['code_lang'] != 'unknown' else []), axis=1)
        return self
        
    def count_code_langs(self):
        """Counts the occurrences of each code_lang, omitting 'unknown'."""
        return Counter(lang for lang in self.df['code_lang'] if lang != 'unknown')

    def count_keywords(self, code_lang_counts, blacklist):
        """Counts the occurrences of non-code-lang keywords, excluding code_langs and blacklist words."""
        all_keywords = []

        # Aggregate all keywords
        for keywords_list in self.df['keywords']:
            all_keywords.extend(keywords_list)

        # Get the set of code_langs to exclude them from keyword counts
        code_langs_set = set(code_lang_counts.keys())

        # Count occurrences of each keyword, excluding 'unknown', code_langs, and blacklist words
        keyword_counts = Counter(keyword for keyword in all_keywords if keyword != 'unknown' and keyword not in code_langs_set and keyword not in blacklist)

        return keyword_counts

    def filter_code_langs(self, code_lang_counts):
        """Filters code_langs based on a dynamically calculated acceptance threshold."""
        counts = np.array(list(code_lang_counts.values()))
        threshold = np.mean(counts)

        filtered_code_lang_counts = {lang: count for lang, count in code_lang_counts.items() if count >= threshold}
        return filtered_code_lang_counts

    def find_top_classes(self, top_n=10):
        blacklist = {'code', 'data', 'example', 'file', 'files', 'github', 'way', 'fixes'}
        
        # Count code_langs
        code_lang_counts = self.count_code_langs()
        
        # Filter code_langs based on the dynamically calculated threshold
        filtered_code_lang_counts = self.filter_code_langs(code_lang_counts)
        
        # Count keywords excluding code_langs and blacklist
        keyword_counts = self.count_keywords(filtered_code_lang_counts, blacklist)
        
        # Get the top N non-code-lang keywords
        top_keywords = keyword_counts.most_common(top_n)
        
        # Combine top keywords and filtered code_langs
        relevant_classes = set(keyword for keyword, _ in top_keywords) | set(filtered_code_lang_counts.keys())
        
        return relevant_classes

    def add_class_column(self, relevant_classes):
        """Adds a 'class' column to the DataFrame, assigning a class if present in keywords."""
        def find_class(keywords):
            for keyword in keywords:
                if keyword in relevant_classes:
                    return keyword
            return None

        self.df['class'] = self.df['keywords'].apply(find_class)
    
    def extract_classes(self, top_n=10):
        relevant_classes = self.find_top_classes(top_n)
        self.add_class_column(relevant_classes)
        print(relevant_classes)
        return self
    
    def filter_by_class(self):
        self.df = self.df[self.df['class'].notnull()]
        
    def remove_elements_missing_class(self):
        self.filter_by_class()
        