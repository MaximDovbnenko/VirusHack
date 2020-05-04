import json

config = {
    #Neural network configuration
    'activation'    : 'SigmoidFunction',
    'alpha'         : 0.4,
    'layers'        : [4, 4],
    'inputs'        : 20,
    'max_iteration' : 200,
    'eps'           : 5,
    'rate'          : 0.5,
    'path_to_train' : 'train_data/neural_dataset',
    #convert 
    'path_to_string'   :   'train_data/string_vectors_train',
    'path_to_out_data' :   'train_data/neural_dataset',
    'path_to_out_test' :   'train_data/neural_dataset_test',
    #word2vec learn configuration
    'default_path_train_data'   : 'train_data/doc2vec_train_data.txt',
    'default_models_path'       : 'out_models/doc2vec_model',
    'epochs'        : 5,
    'vector_size'   : 20,
    #result config
    'out_index_file': 'result/temp.html'

}

def create():
    out_config = open('config/global_config', 'w')
    out_config.write(json.dumps(config, sort_keys=True, indent=4))

if __name__ == "__main__":
    create()