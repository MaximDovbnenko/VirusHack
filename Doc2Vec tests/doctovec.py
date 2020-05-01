from gensim.utils import simple_preprocess
from gensim import corpora
from smart_open   import smart_open
#import nltk
#nltk.download('stopwords')
import gensim
import gensim.downloader as api

import os


class BoWCorpus(object):
    def __init__(self, path, dictionary):
        self.filepath   = path
        self.dictionary = dictionary
    
    def __iter__(self):
        global mydict
        for line in smart_open(self.filepath, encoding="utf-8"):
            tokenized_list = simple_preprocess(line, deacc=True)
            bow = self.dictionary.doc2bow(tokenized_list, allow_update=True)
            mydict.merge_with(self.dictionary)
            yield bow

def load_data_from_file(file_path):
    ret_array = []
    file = open(file_path, encoding='utf-8')
    for line in file:
        tokens = line.split('.')
        for token in tokens: 
            ret_array.append(token)
    return ret_array

def create_tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])



#train_data = list(create_tagged_document(data))   

default_data_path = "data/data.txt"
data = load_data_from_file(default_data_path)
train_data = list(create_tagged_document(data))   
print(train_data)

'''
mydict = corpora.Dictionary()
bow_corpus = BoWCorpus(default_data_path, dictionary = mydict)

'''

model = gensim.models.doc2vec.Doc2Vec(vector_size = 20, min_count = 2, epochs = 140)
model.build_vocab(train_data)

model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)

print(model.infer_vector(['искомых']))
#dataset = api.load("data/data.txt")

#print(dictionary.token2id)

