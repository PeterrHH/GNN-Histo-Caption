'''
Only Run this once, when data is extracted
'''
import os
import argparse
import yaml
import json
import shutil
from glob import glob
import random

#   Basic Setup
with open("config/config.yaml", 'r') as yaml_file:
    config_data = yaml.safe_load(yaml_file)

dataset_path = config_data["dataset_path"]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type = str, 
        default = "../../../../../../srv/scratch/bic/peter/Report", help = "Path to Dataset")
args = parser.parse_args()
if args.dataset_path:
    dataset_path = args.dataset_path

train_anno_path = os.path.join(dataset_path,"train_annotation.json")
test_anno_path = os.path.join(dataset_path,"test_annotation.json")
eval_anno_path = os.path.join(dataset_path,"eval_annotation.json")

full_img_path = os.path.join(dataset_path,"Images")

train_img_folder_path = os.path.join(dataset_path,"Images/train")
test_img_folder_path = os.path.join(dataset_path,"Images/test")
eval_img_folder_path = os.path.join(dataset_path,"Images/eval")
print(train_img_folder_path)
all_img_name = glob(os.path.join(full_img_path,"*png"))
# print(len(all_img_name))
# print(all_img_name[0:2])
# for i in all_img_name[0:2]:
#     print(os.path.splitext(os.path.basename(i))[0])

''' 
Extract image from train_annotations.json into train folder
'''
#   Extract all train name
with open(train_anno_path, 'r') as json_file:
    train_anno = json.load(json_file)

train_name = list(train_anno.keys())

if os.path.exists(train_img_folder_path):
    print("exist")
else:
    print("not")
Count = 0
#   Move them into train folder 
if not os.path.exists(train_img_folder_path):
    print("Here")
    os.makedirs(train_img_folder_path)

    for train_img in train_name:
        source_path = os.path.join(full_img_path,train_img+".png")
        destination_path = os.path.join(train_img_folder_path,train_img+".png")
        if os.path.exists(source_path):
            shutil.move(source_path,destination_path)
 
all_img_name_after = glob(os.path.join(full_img_path,"*png"))
print(len(all_img_name_after))
print(f"{len(all_img_name_after)} images in original /Images folder after processing train image")


''' 
Extract some test_annoations into eval_annotaions, 889 evaluation, 1000 testing
'''
print("Split annotaion between evaluation and testing")
with open(test_anno_path, 'r') as json_file:
    test_anno = json.load(json_file)
if len(list(test_anno.keys())) == 1889:

    # Set the seed for reproducibility
    random.seed(42)

    test_anno_items = list(test_anno.items())
    random.shuffle(test_anno_items)

    # Split the data into two parts
    evaluation_data = test_anno_items[:889]
    testing_data = test_anno_items[889:]

    evaluation_data = dict(evaluation_data)
    testing_data = dict(testing_data)



    #  Move evaluation and rewirte test annotaiton
    with open(eval_anno_path,'w') as write_json_file:
        json.dump(evaluation_data, write_json_file, indent=4)


    with open(test_anno_path, 'w') as write_json_file:
        json.dump(testing_data, write_json_file, indent = 4)


''' 
Extract image from test_annotaions into train and eval image folders accordingly
'''
#   Testing file
with open(test_anno_path, 'r') as json_file:
    test_anno = json.load(json_file)
test_name = list(test_anno.keys())
assert len(test_name) == 1000

if not os.path.exists(test_img_folder_path):
    os.makedirs(test_img_folder_path)

    for test_img in test_name:
        source_path = os.path.join(full_img_path,test_img+".png")
        destination_path = os.path.join(test_img_folder_path,test_img+".png")
        if os.path.exists(source_path):
            shutil.move(source_path,destination_path)
 
all_img_name_after = glob(os.path.join(full_img_path,"*png"))
print(f"{len(all_img_name_after)} images in original /Images folder after processing test image")




#   Evaluation File
with open(eval_anno_path, 'r') as json_file:
    eval_anno = json.load(json_file)
eval_name = list(eval_anno.keys())
assert len(eval_name) == 889

if not os.path.exists(eval_img_folder_path):
    os.makedirs(eval_img_folder_path)

    for eval_img in eval_name:
        source_path = os.path.join(full_img_path,eval_img+".png")
        destination_path = os.path.join(eval_img_folder_path,eval_img+".png")
        if os.path.exists(source_path):
            shutil.move(source_path,destination_path)
 
all_img_name_after = glob(os.path.join(full_img_path,"*png"))
print(f"{len(all_img_name_after)} images in original /Images folder after processing test image")

