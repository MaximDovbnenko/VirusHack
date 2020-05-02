from gensim.utils import simple_preprocess
from gensim import corpora
from smart_open   import smart_open
import gensim
import gensim.downloader as api
import json

class ConvertStringToVector:
        def __init__(self):
            self.PathToFile  = 'train_data/string_vectors_train'
            self.PathToModel = 'out_models/doc2vec_model'
            self.Doc2VecModel = gensim.models.doc2vec.Doc2Vec.load(self.PathToModel)
            #vector = self.Doc2VecModel.infer_vector(["system", "response"])
            #print(vector)
            self.Serialize()
            pass
        def Serialize(self):
            out_result = []
            model_file = open(self.PathToFile , 'r')
            data_model = model_file.read()
            json_model = json.loads(data_model) 
            for line in json_model:
                tmp_string_array = line[0].split(' ')
                vector = self.Doc2VecModel.infer_vector(tmp_string_array)
                list_vector = []
                for val in vector:
                    list_vector.append(float(val))
                out_result.append(
                    [list_vector, line[1]]
                )
            file = open("train_data/xor_test", 'w')
            file.write(json.dumps(out_result, sort_keys=True, indent=2))