# from DatasetParser import DatasetParser
# from bert_classifier import main
from GraphParser import GraphParser

# parser = DatasetParser('dataset', 'dataset_final2.csv')
# parser.read_and_merge_parquet(1, 128, 256)
# parser.parse_content()
# parser.extract_keywords_and_code()
# # parser.load_csv('dataset_final.csv')
# parser.extract_classes(24, 24)
# parser.create_csv('dataset_final2.csv')
# parser.remove_elements_missing_class()
# df = parser.get_df()
# print(df.info())
# print(df.head()) 
# parser.create_csv('dataset_final2_classes.csv') 
# parser.prune_data()
# df = parser.get_df()
# print(df.info())
# print(df.head()) 
# parser.create_csv('dataset_final2_classes_pruned.csv')


# main('dataset_001_classed.csv', '001_bert', 20)
GraphParser.main('dataset_final2_classes_pruned.csv');