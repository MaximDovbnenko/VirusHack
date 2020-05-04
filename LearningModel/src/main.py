from TrainDoc2Vec import *
from NeuralNetTrain import *
from ConvertStringToVector import *
import click
import json

global_config = {}

#####################

@click.command()
@click.argument('operation')
def main(operation):
    config_file   = open('config/global_config' , 'r')
    cohfig_data   = config_file.read()
    global_config = json.loads(cohfig_data) 
    if(operation == 'doc2vec_train'):
        cDoc2VecTrain = Doc2VecTrainModel(global_config)
        cDoc2VecTrain.TrainModel()
    elif operation == 'nn_train':
        TestLearningNet(global_config)
    elif operation == 'convert':
        ConvertString = ConvertStringToVector(global_config)



def TestLearningNet(config):
    NN_TrainData = LoadTrainPattern(config['path_to_train'])
    NN_TestData  = LoadTrainPattern(config['path_to_out_test'])
    NN_Object    = NeuralNetTrain(config['layers'], config['inputs'], config['alpha'], config['rate'], config['max_iteration'] , config['eps'],  NN_TrainData)
    NN_Object.TrainModel()
    count = 0
    good = 0
    bad  = 0
    for i in NN_TestData.InputList:
        Answer = NN_Object.NeuralNet.CalculateOutT(i, 0.5)
        count = 0
        result = 0
        train_namber = 0
        for ans in Answer:
            result += (ans - NN_TestData.OutputList[train_namber][count])
            count+=1
        if result == 0:
            good += 1
        else:
            bad  += 1
        train_namber += 1
    lenght_test = len(NN_TestData.InputList)
    percent = (good / lenght_test) * 100
    print("good is " + str(good) + " bad is " + str(bad))
    print("mode percent recognition = " + str(percent) + " % ")
       
    pass

if __name__ == '__main__':
    main()
