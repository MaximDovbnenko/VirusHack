

config = {
    #Neural network configuration
    'activation'    : 'SigmoidFunction',
    'alpha'         : 0.9
    'layers'        : [2, 4, 1],
    'inputs'        : 3,
    'max_iteration' : 2000000,
    'eps'           : 0.01,
    'path_to_train' : 'train_data/neural_dataset',
    #word2vec learn configuration
    'path_to_dic'   : 'train_data/doc2vec_train_data.txt',
    'epochs'        : 100,
    'vector_size'   : 20,

}