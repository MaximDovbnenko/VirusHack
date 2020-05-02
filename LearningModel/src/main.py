from TrainDoc2Vec import *
from NeuralNetTrain import *
from ConvertStringToVector import *
import click



#####################

@click.command()
@click.argument('operation')
def main(operation):
    if(operation == 'doc2vec_train'):
        cDoc2VecTrain = Doc2VecTrainModel(10, 140)
        cDoc2VecTrain.TrainModel()
    elif operation == 'nn_train':
        TestLearningNet()
    elif operation == 'convert':
        ConvertString = ConvertStringToVector()


def TestLearningNet():
    NN_TrainData = LoadTrainPattern('train_data/xor_test')
    #layer_list, input_size, sigmoid_param, rate, data_train
    NN_Object    = NeuralNetTrain([5, 4, 4], 10, 0.5, 0.9, NN_TrainData)
    NN_Object.TrainModel()
    test = NN_Object.NeuralNet.LoadModel()
    count = 0
    for i in NN_TrainData.InputList:
        Answer = test.CalculateOutT(i, 0.5)
        print(Answer, NN_TrainData.OutputList[count])
        count+=1
    pass

if __name__ == '__main__':
    main()
