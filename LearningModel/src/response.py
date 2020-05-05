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
        self.TemplatePath = config['out_index_file']
        self.TemplateFile = open(self.TemplatePath, 'r')
        self.Context      = self.TemplateFile.read()
        self.process_1 = 0
        self.process_2 = 0
        self.process_3 = 0
        self.process_4 = 0
        self.process_5 = 0
       
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
        if value[0] == 1: 
            self.process_1 += 1
            return  "style='color: black; background:red'"
        elif value[1] == 1: 
            self.process_2 += 1
            return "style='color: black; background:blue'"
        elif value[2] == 1: 
            self.process_3 += 1
            return "style='color: black; background:green'"
        elif value[3] == 1:
            self.process_4 += 1 
            return "style='color: black; background:fuchsia'"
        else: 
            self.process_5 += 1
            return "style='color:black'"
    def Recognize(self):
        out_string = ""
        vec = []
        count_lines = len(self.Data)
        count_token = 0
        for line in self.Data:
            if len(line) > 30:
                list_vector = []
                tmp_token = line.split(' ')
                count_token += len(tmp_token)
                vector = self.Doc2VecModel.infer_vector(line.split(' '))
                for val in vector:
                    list_vector.append(float(val) * 10)
                vec = list_vector
                Answer = self.Net.CalculateOutT(list_vector, 0.5)
                color = self.GetColor(Answer)
                out_string += "<div " + color + ">" + line + "</div>\n"
        print(vec)
        OutFile = open("result/report.html", 'w')
        
        self.process_1 = (self.process_1 / count_lines) * 100
        self.process_2 = (self.process_2 / count_lines) * 100
        self.process_3 = (self.process_3 / count_lines) * 100
        self.process_4 = (self.process_4 / count_lines) * 100
        self.process_5 = (self.process_5 / count_lines) * 100

        self.Context = self.Context.replace("__TEXT__", out_string)
        self.Context = self.Context.replace("TEXT_NAME", self.FileName)
        self.Context = self.Context.replace("COUNT_LINES", str(count_lines))
        self.Context = self.Context.replace("TOKEN_COUNT", str(count_token))
        self.Context = self.Context.replace("PROGRESS_1", "background-color:red ; width: " + str(self.process_1) + "%")
        self.Context = self.Context.replace("PROGRESS_2", "background-color:blue ; width: " + str(self.process_2) + "%")
        self.Context = self.Context.replace("PROGRESS_3", "background-color:green ; width: " + str(self.process_3) + "%")
        self.Context = self.Context.replace("PROGRESS_4", "background-color:fuchsia ; width: " + str(self.process_4) + "%")
        self.Context = self.Context.replace("PROGRESS_5", "background-color:black ; width: " + str(self.process_5) + "%")
        OutFile.write(self.Context)
        print("resul create in result/report.html")
        print("end")

@click.command()
@click.argument('file_name')
def main(file_name):
    config_file   = open('config/global_config' , 'r')
    cohfig_data   = config_file.read()
    global_config = json.loads(cohfig_data) 
    Resp = Response(file_name, global_config)

if __name__ == "__main__":
    main()