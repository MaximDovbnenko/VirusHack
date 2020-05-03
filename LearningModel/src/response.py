from gensim.utils import simple_preprocess
from gensim import corpora
from smart_open   import smart_open
import gensim
import gensim.downloader as api
import json
import click
from MachineLearning import *
import math


class Response:
    def __init__(self, file_name, config):
        self.FileName = file_name
        self.Data = []
        self.PathToModel  = config['default_models_path']
        print(self.PathToModel)
        self.Doc2VecModel = gensim.models.doc2vec.Doc2Vec.load(self.PathToModel)
        self.Net          = NeuralNet(NeuralNet(SigmoidFunction(0), 1, [1,1]))
        self.Net       = self.Net.LoadModel()
        self.SplitData()
        self.Recognize()
        pass
    
    def SplitData(self):
        DataFile = open(self.FileName, 'r')
        all_data = DataFile.read()
        self.Data = all_data.split('.')
        pass
    
    def GetColor(self, value):
        if value[0] == 1: return "style='color:red'"
        elif value[1] == 1: return "style='color:blue'"
        elif value[2] == 1: return "style='color:green'"
        elif value[3] == 1: return "style='color:fuchsia'"
        else: return "style='color:black'"
    def Recognize(self):
        out_string = ""
        vec = []
        for line in self.Data:
            if len(line) > 30:
                list_vector = []
                vector = self.Doc2VecModel.infer_vector(line.split(' '))
                for val in vector:
                    list_vector.append(float(val))
                vec = list_vector
                Answer = self.Net.CalculateOutT(list_vector, 0.5)
                color = self.GetColor(Answer)
                out_string += "<div " + color + ">" + line + "</div>\n"
        print(vec)
        OutFile = open("result/report.html", 'w')
        OutFile.write(out_string)


@click.command()
@click.argument('file_name')
def main(file_name):
    config_file   = open('config/global_config' , 'r')
    cohfig_data   = config_file.read()
    global_config = json.loads(cohfig_data) 
    Resp = Response(file_name, global_config)

if __name__ == "__main__":
    main()