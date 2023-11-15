# import numpy as np
# test_output = {
#     'loss': [3,6],
#     'bleu1':[5,1],
#     'bleu2':[1,9],
#     'bleu3':[2,2],
#     'bleu4':[3,3],
#     'meteor':[6,7],
#     'rouge':[1,0.5],
#     'cider':[0.3,4],
#     'spice':[0.2,0.3],
# }
# for i, key in enumerate(test_output.keys()):
#     print(f"i = {i} key = {key}")
#     test_output[key].append(1)
# print(test_output)


# for key, values in test_output.items():
#     mean_value = np.mean(values)  # Calculate the mean using np.mean
#     std_value = np.std(values)  
#     test_output[key] = [mean_value,std_value] # Store the mean in the new dictionary
# print(test_output)
import os
split = "train"

path = f"../../../../../../srv/scratch/bic/peter/full-graph/tissue_graphs/{split}"

num = len(os.listdir(path))
print(num)

import json

# Replace 'your_file.json' with the path to your JSON file
file_path = f'../../../../../../srv/scratch/bic/peter/Report/{split}_annotation.json'

# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

print(len(data.keys()))
