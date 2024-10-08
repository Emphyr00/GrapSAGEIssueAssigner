from DatasetParser import DatasetParser
# from bert_classifier import main
# from GraphParser import GraphParser
# from GraphModel import train_and_test

thresholds = [
    # [2, 1024, 2048],
    # [10, 512, 1024],
    [8, 24, 48],
]

parser = DatasetParser('dataset', 'dataset_test2.csv', thresholds)
parser.read_and_merge_parquet(0.01)
parser.parse_content()
parser.extract_keywords_and_code()
# parser.load_csv('dataset_final.csv')
parser.extract_classes(24)
parser.create_csv('dataset_test2.csv')
parser.remove_elements_missing_class()
df = parser.get_df()
print(df.info())
print(df.head()) 
parser.create_csv('dataset_test2_classes.csv') 
parser.prune_data()
df = parser.get_df()
print(df.info())
print(df.head()) 
parser.create_csv('dataset_test2_classes_pruned.csv')


# main('dataset_001_classed.csv', '001_bert', 20)
# GraphParser.main('dataset_test2_classes_pruned.csv');
# train_and_test('graph_even_company.gml')