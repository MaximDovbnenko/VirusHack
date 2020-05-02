
import math
import random
import json



class Neuron:

    def __init__(self, inputCount, Activation):
        if inputCount != 0 :
            random.seed(2)
            self.ActivationFunction = Activation
            self.Length = inputCount
            self.Weigth = [(random.random()) * 2 - 0.5  for i in range(self.Length)]
            self.Offset = 1.0
            self.OUTPUT = 0.0

    def Calculate(self, inputVector):
        S = 0.0
        if len(inputVector) == len(self.Weigth):
            for w in range(len(self.Weigth)):
                S += self.Weigth[w] * inputVector[w]
                
            S += self.Offset
            S = self.ActivationFunction.calculate(S)
            self.OUTPUT = S   
        return S


class Layer:


    def __init__(self, NeuronsCount, InputSize, Activation):
        self.Neurons = []
        self.Length = NeuronsCount
        self.OutLayer = [0 for n in range(self.Length)]   

        for neuron in range(NeuronsCount):
            self.Neurons.append(Neuron(InputSize, Activation))
            

    def CalculateOut(self, InputVector):
        for neuron in range(self.Length):
            self.OutLayer[neuron] = self.Neurons[neuron].Calculate(InputVector)
        return self.OutLayer


class NeuralNet:

    
    def __init__(self, Activation = None, InputSize = 0, LayerList = []):
        self.ActivationFunction = Activation
        self.NetInputSize = InputSize
        self.Length = len(LayerList)
        self.LayerList = LayerList
        self.Layers = []
        for layer in range(self.Length):
            if layer == 0 :
                self.Layers.append(Layer(int(LayerList[layer]), InputSize, Activation))
            else:
                self.Layers.append(Layer(int(LayerList[layer]), self.Layers[layer - 1].Length, Activation))


    def CalculateOut(self, inputVector):
        tmp_layer_out = []
        self.OUTPUT = []
        for layer in range(len(self.Layers)):
            if layer == 0 :
                tmp_layer_out = self.Layers[layer].CalculateOut(inputVector)
            else:
                tmp_layer_out = self.Layers[layer].CalculateOut(tmp_layer_out)

        self.OUTPUT = tmp_layer_out
        return tmp_layer_out


    def CalculateOutT(self, inputVector, trheshold):
        tmp_layer_out = []
        self.OUTPUT = []
        for layer in range(len(self.Layers)):
            if layer == 0 :
                tmp_layer_out = self.Layers[layer].CalculateOut(inputVector)
            else:
                tmp_layer_out = self.Layers[layer].CalculateOut(tmp_layer_out)
        for i in range(len(tmp_layer_out)):
            if tmp_layer_out[i] > trheshold:
                tmp_layer_out[i] = 1
            else:
                tmp_layer_out[i] = 0
        self.OUTPUT = tmp_layer_out
        return tmp_layer_out
    def CalculateOutBipolar(self, inputVector):
        tmp_layer_out = []
        self.OUTPUT = []
        for layer in range(len(self.Layers)):
            if layer == 0 :
                tmp_layer_out = self.Layers[layer].CalculateOut(inputVector)
            else:
                tmp_layer_out = self.Layers[layer].CalculateOut(tmp_layer_out)
        for i in range(len(tmp_layer_out)):
            if tmp_layer_out[i] <= 0.5 and tmp_layer_out[i] >= -0.5:
                tmp_layer_out[i] = 0
            elif tmp_layer_out[i] > 0.5:
                tmp_layer_out[i] = 1   
            elif tmp_layer_out[i] < -0.5:
                tmp_layer_out[i] = -1
        self.OUTPUT = tmp_layer_out
        return tmp_layer_out
    def LoadModel(self):
        model_file = open('out_models/neural_net_model', 'r')
        data_model = model_file.read()
        json_model = json.loads(data_model) 
        offset = json_model['offset']
        weigth = json_model['weigth']
        activation = json_model['activation'][0]
        alpha      = float(json_model['activation'][1])
        layers_list = json_model['layers']
        input_size  = json_model['inputs']
        NewNet = None
        if activation == 'SigmoidFunction':
            NewNet = NeuralNet(SigmoidFunction(alpha), input_size, layers_list)
        i = 0
        j = 0
        k = 0
        for layer in NewNet.Layers:
            for neuron in layer.Neurons:
                neuron.Offset = offset[i][j]
                neuron.Weigth = weigth[i][j]
                j += 1
            i += 1
            j = 0
        
        return NewNet
    def setW(self, matrix):
        layers  = matrix[0]
        neurons = matrix[1]
        weidth  = matrix[2]
        LayerList = [layers, neurons, weidth]
        #CurrentNet = NeuralNet(SigmoidFunction(1), )

class SigmoidFunction:    
    def __init__ (self, alpha):
        self.Alpha = alpha
        self.Type  = 'SigmoidFunction'
        
    def calculate(self, value):
        return (1.0 / (1.0 + math.exp(-(value * self.Alpha))))
    
    def derivative(self, value):
        return (self.Alpha * value * (1.0 - value))

class BipolarSigmoidFunction:
    def __init__ (self, alpha):
        self.Alpha = alpha
        self.Type  = 'BipolarSigmoidFunction'
    def calculate(self, value):
        return ( ( 2 / ( 1 + math.exp( -self.Alpha * value ) ) ) - 1 )
    def derivative(self, value):
        return ( self.Alpha  * ( 1 - value * value ) / 2 )
