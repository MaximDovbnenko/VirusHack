from NeuralNET import *
import json
class E_Distance:
    
    def calculate(self, op1, op2):
        self.dist = 0.0
        
        for i in range(len(op1)):
            self.dist += (op1[i] - op2[i]) * (op1[i] - op2[i])
        return 0.5 * self.dist

    def calculateDerivative(self, op1, op2, index):
        return (op1[int(index)] - op2[int(index)])

class BackPropagation:
    def __init__(self, NET, DistanceFunction):
        self.ThisNet   = NET
        self.Momentum  = 0.3      
        self.LearnRate = 0.9
        self.OptimizationFunction = DistanceFunction
        self.neuralNetArray = []
        for i in range(self.ThisNet.Length):
            
             self.neuralNetArray.append([])
             for j in range(self.ThisNet.Layers[i].Length):
                  self.neuralNetArray[i].append([])
                  for k in range(self.ThisNet.Layers[i].Neurons[j].Length):
                      self.neuralNetArray[i][j].append([])
                      self.neuralNetArray[i][j][k] = 0.0
        
        self.ErrorNeuralNet = []
        for i in range(self.ThisNet.Length):
            self.ErrorNeuralNet.append([])
            for j in range(self.ThisNet.Layers[i].Length):
                self.ErrorNeuralNet[i].append(0.0)

        self.Offsets = []
        for i in range(self.ThisNet.Length):
            self.Offsets.append([])
            for j in range(self.ThisNet.Layers[i].Length):
                self.Offsets[i].append(0.0)


    def calculateError(self, TargetPattern):
        self.Activation = self.ThisNet.ActivationFunction
        LayerCount = self.ThisNet.Length
        ILayer = self.ThisNet.Layers[LayerCount - 1]
        tmp_next_layer = None
        ErrorOutNeuron = 0.0


        for neuron in range(ILayer.Length):
            ErrorOutNeuron = \
            self.OptimizationFunction.calculateDerivative(TargetPattern, self.ThisNet.OUTPUT, neuron)
            self.ErrorNeuralNet[LayerCount - 1][neuron] = \
            ErrorOutNeuron * self.Activation.derivative(ILayer.Neurons[neuron].OUTPUT)

        #next_layer_count = LayerCount - 2


        for i in reversed(range(LayerCount - 1)):
            #print("i", i)
            tmp_next_layer = self.ThisNet.Layers[i + 1]
            ILayer = self.ThisNet.Layers[i]
            PriviusError = self.ErrorNeuralNet[i + 1]
            for CurentNeurons in range(ILayer.Length):
                s = 0.0
                for PriviusNeurons in range(tmp_next_layer.Length):
                    s += PriviusError[PriviusNeurons] * \
                         tmp_next_layer.Neurons[PriviusNeurons].Weigth[CurentNeurons] 
                self.ErrorNeuralNet[i][CurentNeurons] = \
                         s * self.Activation.derivative(ILayer.Neurons[CurentNeurons].OUTPUT)   
           
            

    def calculateUpdateWeigth(self, input_vector):
        momentum_ = self.Momentum * self.LearnRate
        MomentumValue = self.LearnRate * (1 - self.Momentum)

      
        for neuron in range(self.ThisNet.Layers[0].Length):
            error = self.ErrorNeuralNet[0][neuron]
            for weigth in range(self.ThisNet.Layers[0].Neurons[neuron].Length):
                self.neuralNetArray[0][neuron][weigth] = momentum_ * self.neuralNetArray[0][neuron][weigth] + \
                self.LearnRate * error * input_vector[weigth]
            self.Offsets[0][neuron] = self.LearnRate * error
       
        for layer in range(1, self.ThisNet.Length):
            for neuron in range(self.ThisNet.Layers[layer].Length):
                error = self.ErrorNeuralNet[layer][neuron]
                for weigth in range(self.ThisNet.Layers[layer].Neurons[neuron].Length):
                    
                    self.neuralNetArray[layer][neuron][weigth] = \
                    momentum_ * self.neuralNetArray[layer][neuron][weigth] + self.LearnRate * error * \
                    self.ThisNet.Layers[layer - 1].Neurons[weigth].OUTPUT
                    
                self.Offsets[layer][neuron] = self.LearnRate * error

    def setNewWeigth(self):
        for layer in range(self.ThisNet.Length):
            for neuron in range(self.ThisNet.Layers[layer].Length):
                for weigth in range(self.ThisNet.Layers[layer].Neurons[neuron].Length):
                    self.ThisNet.Layers[layer].Neurons[neuron].Weigth[weigth] += \
                                        self.neuralNetArray[layer][neuron][weigth]
                self.ThisNet.Layers[layer].Neurons[neuron].Offset += self.Offsets[layer][neuron]

    def startProcess(self, maxItr, MaxError, InputPatterns, OutputPatterns):
        Error = 0.0
        countPrint = 0
        Itr = 0
        ErrorLog = open('neural_net_log/net_error.log', 'w')
        for itr in range(maxItr):
            Error = 0.0
            for pattern in range(len(InputPatterns)):
                NetAnswer = self.ThisNet.CalculateOut(InputPatterns[pattern])  
                Error += self.OptimizationFunction.calculate(OutputPatterns[pattern], NetAnswer)
                self.calculateError(OutputPatterns[pattern])
                self.calculateUpdateWeigth(InputPatterns[pattern])
                self.setNewWeigth()
            if countPrint > 500:
                print(Itr, Error)
                countPrint = 0
            else:
                countPrint += 1
            Itr+=1
            ErrorLog.write(str(itr) + " " + str(Error) + "\n")
            if Error < MaxError: break
        self.Save()
        ErrorLog.close()
    def Save(self):
        for layer in range(self.ThisNet.Length):
            for neuron in range(self.ThisNet.Layers[layer].Length):
                for weigth in range(self.ThisNet.Layers[layer].Neurons[neuron].Length):
                    self.neuralNetArray[layer][neuron][weigth] = self.ThisNet.Layers[layer].Neurons[neuron].Weigth[weigth] 
                self.Offsets[layer][neuron] = self.ThisNet.Layers[layer].Neurons[neuron].Offset
        
        Model = {
            'offset' : self.Offsets,
            'weigth' : self.neuralNetArray,
            'inputs' : self.ThisNet.NetInputSize,
            'layers' : self.ThisNet.LayerList,
            'activation' : [ 
                self.ThisNet.ActivationFunction.Type,
                self.ThisNet.ActivationFunction.Alpha
            ]
            
        }
        FileModel = open('out_models/neural_net_model', 'w')
        FileModel.write(json.dumps(Model, sort_keys=True, indent=4))

        
    



