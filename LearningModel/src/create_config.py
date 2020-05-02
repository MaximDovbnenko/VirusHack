import json

config = {
    #Neural network configuration
    'activation'    : 'SigmoidFunction',
    'alpha'         : 0.5,
    'layers'        : [ 5, 4],
    'inputs'        : 10,
    'max_iteration' : 2000000,
    'eps'           : 15,
    'rate'          : 0.5,
    'path_to_train' : 'train_data/neural_dataset',
    #convert 
    'path_to_string'   :   'train_data/string_vectors_train',
    'path_to_out_data' :   'train_data/neural_dataset',
    'path_to_out_test' :   'train_data/neural_dataset_test',
    #word2vec learn configuration
    'default_path_train_data'   : 'train_data/doc2vec_train_data.txt',
    'default_models_path'       : 'out_models/doc2vec_model',
    'epochs'        : 100,
    'vector_size'   : 10

}

def create():
    out_config = open('config/global_config', 'w')
    out_config.write(json.dumps(config, sort_keys=True, indent=4))

if __name__ == "__main__":
    create()