from gensim.utils import simple_preprocess
from gensim import corpora
from smart_open   import smart_open
import gensim
import gensim.downloader as api

import os

class Doc2VecTrainModel:
    def __init__(self, config):
        self.default_path_train_data = config['default_path_train_data']
        self.default_models_path     = config['default_models_path']
        self.TrainData = []
        self.CreateTokenList()
        self.out_vector_size = config['vector_size']
        self.max_epochs      = config['epochs']
        pass
    def create_tagged_document(self, list_of_list_of_words):
        for i, list_of_words in enumerate(list_of_list_of_words):
            yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

    def CreateTokenList(self):
        print("creating token list from " + self.default_path_train_data)
        with open(self.default_path_train_data , 'r') as input_file:
            data = input_file.read()
        split_data = data.split('.')
        tmp_data = []
        for line in split_data:
            if len(line) > 30:
                tmp_data.append(line)
        print("create " + str(len(tmp_data)) + " lines")
        print("convert data in doc2vec format...")
        self.TrainData = list(self.create_tagged_document(tmp_data))
        #print(self.TrainData)
        pass

    def TrainModel(self):
        print("strat doc2vec train ( vector size " + str(self.out_vector_size) + " epochs " + str(self.max_epochs) + " )")
        self.Doc2VecModel = gensim.models.doc2vec.Doc2Vec(vector_size = self.out_vector_size, min_count = 2, epochs = self.max_epochs )
        self.Doc2VecModel.build_vocab(self.TrainData)
        self.Doc2VecModel.train(self.TrainData, total_examples=self.Doc2VecModel.corpus_count, epochs=self.Doc2VecModel.epochs)
        print("end doc2vec train...")
        print("save model in " + self.default_models_path)
        self.SaveModel()
        pass
    
    def SaveModel(self):
        self.Doc2VecModel.save(self.default_models_path)
        pass

class Doc2VecResponse:
    def __init__(self):
        pass