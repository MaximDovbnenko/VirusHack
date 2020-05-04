from gensim.utils import simple_preprocess
from gensim import corpora
from smart_open   import smart_open
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import gensim
import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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

            for t in token.split(' '):
                ret_array.append(t)
    return ret_array

data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

max_epochs = 100
vec_size = 20
alpha = 0.025
'''
model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=2,
                dm =1,
                workers=1,
                negative=0)
  
model.build_vocab(tagged_data)
model.random.seed(0)
for epoch in range(max_epochs):
    #print('iteration {0}'.format(epoch))
    model.random.seed(0)
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
test_data = word_tokenize("I love chatbots".lower())
model.save("d2v.model")
print("Model Saved")
model.random.seed(42)

v1 = model.infer_vector(test_data)
print(v1)
'''
model= Doc2Vec.load("d2v.model")
model.init_sims(replace=True)
#to find the vector of a document which is not in training data
test_data = word_tokenize("I love chatbots".lower())
print(test_data)
model.random.seed(42)
for i in range(10):
    v1 = model.infer_vector(test_data)
    print(v1)
print(v1)


