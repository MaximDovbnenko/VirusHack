from MachineLearning import *
import math


class LoadTrainPattern:
    def __init__(self, path_to_train = "train_data/neural_net_train"):
        self.InputList  = []
        self.OutputList = []
        self.PathToTrain = path_to_train
        self.LoadFile()
        pass
    def LoadFile(self):
        data_file = open(self.PathToTrain, 'r')
        data = data_file.read()
        json_data = json.loads(data) 
        for data_line in json_data:
            self.InputList.append(data_line[0])
            self.OutputList.append(data_line[1])
           # print(data_file[0], data_file[1])
        for t in self.OutputList:
            print(t)
        pass

class NeuralNetTrain:
    def __init__(self, layer_list, input_size, sigmoid_param, rate, data_train):
        self.LearningRate = rate
        self.DataTrain = data_train
        self.LayerList = layer_list
        self.InputSize = input_size
        self.SigmoidParam = sigmoid_param
        self.NeuralNet = NeuralNet(SigmoidFunction(self.SigmoidParam), self.InputSize, self.LayerList)
        self.BP = BackPropagation(self.NeuralNet, E_Distance())
        self.BP.LearnRate =  self.LearningRate 
        pass

    def TrainModel(self):
        self.BP.startProcess(200000,0.1,self.DataTrain.InputList, self.DataTrain.OutputList)
        pass

    def SaveModel(self):
        pass