import os
import pandas as pd
import re
import fasttext
import fasttext.util
import spacy
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from guesslang import Guess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import ast
import inflect
from IPython.display import display
import matplotlib.pyplot as plt
from textwrap import wrap

inflect_engine = inflect.engine()
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
fasttext.util.download_model('en', if_exists='ignore')
fasttext_model = fasttext.load_model('cc.en.300.bin')

class DatasetParser:
    def __init__(self, directory_path, output_csv, thresholds):
        self.directory_path = directory_path
        self.output_csv = output_csv
        self.df = None
        self.guess = Guess()
        self.thresholds = thresholds 

    def create_csv(self, output_file=None):
        self.df.to_csv(output_file or self.output_csv, index=False, escapechar='\\')
        return self

    def get_df(self):
        return self.df

    def display_sample(self, step_name, row_index=0, max_colwidth=30, max_lines_per_cell=5):
        print(f"Sample data for row {row_index} after step: {step_name}")

        if row_index >= len(self.df):
            print(f"Error: The row_index {row_index} is out of range. Defaulting to row 0.")
            row_index = 0
        sample = self.df.iloc[[row_index]]  # Always take the same element

        filename = f"{step_name}_row_{row_index}.csv"
        sample.to_csv('datachange/' + filename, index=False)

        print(f"Saved sample data to {filename}")
        
    def read_and_merge_parquet(self, size=1):
        print('read_and_merge_parquet')
        data_frames = []
        
        files = [file_name for file_name in os.listdir(self.directory_path) if file_name.endswith('.parquet')]
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(pd.read_parquet, os.path.join(self.directory_path, file_name)): file_name for file_name in files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Reading Parquet Files"):
                df = future.result()
                df = df[['repo', 'content']]
                
                if 0 < size < 1:
                    df = df.sample(frac=size)
                
                data_frames.append(df)

        self.df = pd.concat(data_frames, ignore_index=True)
        self.filter_repos_by_thresholds()
        
        self.display_sample("read_and_merge_parquet")
        return self

    def filter_repos_by_thresholds(self):
        all_filtered_dfs = []

        for number, lower_threshold, upper_threshold in self.thresholds:
            repo_counts = self.df['repo'].value_counts()
            filtered_repos = repo_counts[(repo_counts >= lower_threshold) & (repo_counts <= upper_threshold)].index
            filtered_df = self.df[self.df['repo'].isin(filtered_repos)]

            if len(filtered_repos) > number:
                selected_repos = filtered_repos.to_series().sample(n=number, random_state=42).index
            else:
                selected_repos = filtered_repos

            filtered_df = filtered_df[filtered_df['repo'].isin(selected_repos)]
            all_filtered_dfs.append(filtered_df)

        self.df = pd.concat(all_filtered_dfs, ignore_index=True)
        self.print_unique_repo_counts()

        self.display_sample("filter_repos_by_thresholds")

    def print_unique_repo_counts(self):
        print("Unique repo values with their counts (sorted by number of samples, descending):")
        repo_counts = self.df['repo'].value_counts().sort_values(ascending=False)
        for repo, count in repo_counts.items():
            print(f"{repo}: {count}")

    def clean_and_extract_content(self, text):
        cleaned_text = re.sub(r'<issue_closed>', '', text)
        cleaned_text = re.sub(r'username_\d+:', '', cleaned_text)
        cleaned_text = re.sub(r'@username_\d', '', cleaned_text)
        cleaned_text = re.sub(r'!\[.*?\]\(.*?\)', '', cleaned_text)
        
        code_blocks = re.findall(r'`{1,3}(.*?)`{1,3}', cleaned_text, re.S)
        code = ' '.join(code_blocks).strip()
        
        cleaned_text = re.sub(r'`{1,3}.*?`{1,3}', '', cleaned_text, flags=re.S)
        
        matches = re.search(r'<issue_comment>(.*?)(?:<issue_comment>|$)', cleaned_text, re.S)
        if matches:
            content = matches.group(1).strip()
            content = re.sub(r'^Title:\s*', '', content)
            content = content.replace('\n', ' ').replace('\r', ' ')
            content = re.sub(r'\s+', ' ', content)
            return content, code

        return "", code

    def parse_content(self):
        print('parse_content')
        tqdm.pandas(desc="Parsing Content")
        self.df[['extracted_content', 'code']] = self.df['content'].progress_apply(lambda text: pd.Series(self.clean_and_extract_content(text)))
        self.df = self.df[['repo', 'extracted_content', 'code']]
        
        self.display_sample("parse_content")
        return self

    def get_keywords_fasttext(self, text, model, top_n=5):
        if pd.isna(text):
            return []

        doc = nlp(text.lower())
        words = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN') and token.is_alpha and token.text not in stop_words]
        
        singular_words = [inflect_engine.singular_noun(word) or word for word in words]

        word_vectors = {word: model.get_word_vector(word) for word in singular_words}
        
        if not word_vectors:
            return []

        centroid = np.mean(list(word_vectors.values()), axis=0)

        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        similarities = {word: cosine_similarity(vector, centroid) for word, vector in word_vectors.items()}
        sorted_words = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        keywords = [word for word, _ in sorted_words[:top_n]]

        return keywords

    def detect_language(self, code):
        if not code or pd.isna(code):
            return 'unknown'

        try:
            lang = self.guess.language_name(code)
        except:
            lang = 'unknown'
        
        if lang is None or lang.lower() == 'text' or lang.lower() == 'ini':
            return 'unknown'
        
        return lang.lower()

    def extract_keywords_and_code(self):
        print('extract_keywords_and_code')
        tqdm.pandas(desc="Extracting Keywords and Code")
        self.df['code_lang'] = self.df['code'].progress_apply(self.detect_language)
        self.df['keywords'] = self.df.progress_apply(lambda row: self.get_keywords_fasttext(row['extracted_content'], fasttext_model, top_n=5) + ([row['code_lang']] if row['code_lang'] != 'unknown' else []), axis=1)
        
        self.display_sample("extract_keywords_and_code")
        return self

    def count_code_langs(self):
        return Counter(lang for lang in self.df['code_lang'] if lang != 'unknown')

    def count_keywords(self, code_lang_counts, blacklist):
        all_keywords = []
        for keywords_list in self.df['keywords']:
            all_keywords.extend(keywords_list)

        code_langs_set = set(code_lang_counts.keys())
        keyword_counts = Counter(keyword for keyword in all_keywords if keyword != 'unknown' and keyword not in code_langs_set and keyword not in blacklist)
        return keyword_counts

    def filter_code_langs(self, code_lang_counts):
        filtered_code_lang_counts = {lang: count for lang, count in code_lang_counts.items() if count >= 50}
        return filtered_code_lang_counts

    def find_top_classes(self, top_n=10):
        blacklist = set(stopwords.words('english')).union({'code', 'data', 'example', 'file', 'files', 'github', 'way', 'fixes', 'fix', 'issue', 'pr', 'test', 'work', 'change', 'changes', 'issues', 'kind', 'use', 'error', 'make', 'description'})
        code_lang_counts = self.count_code_langs()
        filtered_code_lang_counts = self.filter_code_langs(code_lang_counts)
        keyword_counts = self.count_keywords(filtered_code_lang_counts, blacklist)
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.df['extracted_content'].dropna())
        feature_names = vectorizer.get_feature_names_out()
        
        tfidf_scores = zip(feature_names, tfidf_matrix.sum(axis=0).tolist()[0])
        sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        
        top_keywords = [(keyword, score) for keyword, score in sorted_keywords if keyword not in blacklist and keyword in keyword_counts]
        print(top_keywords[:top_n])
        
        relevant_classes = set(keyword for keyword, _ in top_keywords[:top_n]) | set(filtered_code_lang_counts.keys())
        return relevant_classes

    def add_class_column(self, relevant_classes):
        def find_class(keywords):
            for keyword in keywords:
                if keyword in relevant_classes:
                    return keyword
            return None

        self.df['class'] = self.df['keywords'].apply(find_class)

    def extract_classes(self, top_n=10):
        print('extract_classes')
        relevant_classes = self.find_top_classes(top_n)
        self.add_class_column(relevant_classes)
        print(relevant_classes)

        self.display_sample("extract_classes")
        return self

    def filter_by_class(self):
        self.df = self.df[self.df['class'].notnull()]

    def remove_elements_missing_class(self):
        print('remove_elements_missing_class')
        self.filter_by_class()
        
        self.display_sample("remove_elements_missing_class")
        return self

    def prune_data(self):
        self.df = self.df.drop(columns=['code', 'extracted_content'])
        
        self.display_sample("prune_data")
        return self

    def load_csv(self, file_path, keywords_column='keywords'):
        self.df = pd.read_csv(file_path)
        if keywords_column in self.df.columns:
            self.df[keywords_column] = self.df[keywords_column].apply(self._convert_to_list)
        return self

    def _convert_to_list(self, item):
        if isinstance(item, str):
            try:
                return ast.literal_eval(item)
            except (ValueError, SyntaxError):
                return [item]
        return item
