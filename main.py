from DatasetParser import DatasetParser

parser = DatasetParser('dataset', 'test.csv')
parser.read_and_merge_parquet(0.001)
parser.parse_content()
parser.extract_keywords_and_code()
parser.extract_classes(10)
parser.create_csv('dataset_classed_all.csv')
parser.remove_elements_missing_class()
df = parser.get_df()
print(df.info())
print(df.head())
parser.create_csv('dataset_classed_filterd.csv')