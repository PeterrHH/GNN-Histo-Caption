import glob

import os
# Specify the path to the folder you want to count files in
folder_path = 'full-graph/cell_graphs/eval/'

# Use glob to get a list of all files in the folder
file_list = glob.glob(os.path.join(folder_path,"*"))

# Print the total count of files in the folder
print(f'Total number of files in the folder: {len(file_list)}')