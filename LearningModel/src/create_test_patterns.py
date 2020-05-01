import json

data = [
    [[0,0], [0]],
    [[0,1], [1]],
    [[1,0], [1]],
    [[1,1], [0]]
]

file = open("train_data/xor_test", 'w')
file.write(json.dumps(data, sort_keys=True, indent=2))
