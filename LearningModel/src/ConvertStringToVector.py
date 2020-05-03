from gensim.utils import simple_preprocess
from gensim import corpora
from smart_open   import smart_open
import gensim
import gensim.downloader as api
import json

class ConvertStringToVector:
        def __init__(self, config):
            self.PathToFile  = config['path_to_string']
            self.PathToModel = config['default_models_path']
            self.OutData     = config['path_to_out_data']
            self.OutTest     = config['path_to_out_test']
            self.Doc2VecModel = gensim.models.doc2vec.Doc2Vec.load(self.PathToModel)
            #vector = self.Doc2VecModel.infer_vector(["system", "response"])
            #print(vector)
            self.Serialize()
            pass
        def Serialize(self):
            print("OK")
            out_result = []
            out_test   = []
            model_file = open(self.PathToFile , 'r', encoding="utf-8")
            data_model = model_file.read()
            json_model = json.loads(data_model) 
            length = len(json_model)
            print("start convert.." + str(length) + " strings")
            train_len = int((length / 100) * 70)
            count = 0
            for line in json_model:
                tmp_string_array = line[0].split(' ')
                tmp_array_filtred = []
                for val in tmp_string_array:
                    if val != '':
                        tmp_array_filtred.append(val)
               
                vector = self.Doc2VecModel.infer_vector(list(tmp_array_filtred))
                print(vector)
                list_vector = []
                for val in vector:
                    list_vector.append(float(val))
                    if count <= train_len:
                        out_result.append(
                            [list_vector, line[1]]
                        )
                    else:
                        out_test.append(
                            [list_vector, line[1]]
                        )
                count += 1
            file = open(self.OutData  , 'w')
            file.write(json.dumps(out_result, sort_keys=True, indent=2))
            file = open(self.OutTest  , 'w')
            file.write(json.dumps(out_test, sort_keys=True, indent=2))
            print("end")
            