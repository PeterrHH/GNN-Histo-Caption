import os
from simple_parsing import ArgumentParser
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import math
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import WeightedRandomSampler

import yaml
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from evaluation import Scorer
from dataloader import make_dataloader, dataset_to_loader

import wandb

from argparse import ArgumentParser
from models.Attention import EncoderLayer
from Vocabulary import Vocabulary
from models.Graph_Model import GNNEncoder
from models.LSTM import LSTMDecoder
from models.GlobalFeatureExtractor import GlobalFeatureExtractor
from models.Classifier import Classifier
from models.Transformer import TransformerDecoder
from data_plotting import violin_plot

def update_argparser(args, config_file_path = "config/config.yaml"):
    if args.config_path:
        config_file_path = args.config_path
    with open(config_file_path, "r") as conf_file:
        config = yaml.full_load(conf_file)
    
    if args.phase:
        config["phase"] = args.phase
    if args.in_ram:
        config["in_ram"] = True
    else:
        config["in_ram"] = False
    if args.epochs:
        config["epochs"] = args.epoch
    print(config)
    return config

'''
Parse Argument
'''
def argparser():
    parser = ArgumentParser()

    parser.add_argument(
        '--phase',
        help = "Enter either: train, test or eval, represent the stage of model",
        type = str,
        choices = ["train","test","eval"]
        )

    parser.add_argument('--config_path',
        help = "path to the config path used",
        type = str)

    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )

    parser.add_argument(
        '--epochs', type=int, help='epochs.', required=False
    )

    parser.add_argument(
        '-lr','--learning_rate', type = float, help = "set learning rate"
    )

    return parser
'''

WEighted Sample
'''
def get_sample_samplier(dataset,batch_size):
    class_count = {
        '0': 0,
        '1': 0,
        '2': 0,
        }

    count = 0
    class_weight = []
    for idx,output in enumerate(dataset):
        _, _, _, _, labels, _, _= output
        class_count[str(labels)] += 1
        count += 1

    print(class_count)
    for key,value in class_count.items():
        class_weight.append(1/value)
    print(class_weight)
    sample_weights = [0]*len(dataset)

    for idx, output in enumerate(dataset):
        _, _, _, _, labels, _, _= output
        # print(labels.shape)
        class_count[str(labels)] += 1
        sample_weights[idx] = class_weight[labels]
    sampler = WeightedRandomSampler(weights = sample_weights,num_samples = len(dataset),replacement = True)
    dl = dataset_to_loader(dataset, sampler = sampler, batch_size = batch_size)
    return dl

def get_all_models(args,device): 
    # current_directory = os.getcwd()

    encoder_path = os.path.join(args["model_save_base_path"],args["load_encoder_name"])
    global_feat_path = os.path.join(args["model_save_base_path"],args["load_global_extractor_name"])
    # print(f"current dir {current_directory}")
    # print(f"encoder path {encoder_path} and {os.path.exists(encoder_path)}")
    # print(f"global feat path {global_feat_path} and {os.path.exists(global_feat_path)} ")
    # files = os.listdir(args["model_save_base_path"])
    # if args["load_encoder_name"] in files:
    #     print(f"{args['load_encoder_name']} in directory")

    # if args["load_global_extractor_name"] in files:
    #     print(f"{args['load_global_extractor_name']} in directory")
    # for i in files:
    #     print(i)


    if os.path.exists(encoder_path):
        # load
        print(f"exists")
        encoder = torch.load(encoder_path).to(device)
        pass
    else:
        raise Exception(f"Encoder path {encoder_path} not exist")
    
    if os.path.exists(global_feat_path):
        # load
        global_feat_extractor = torch.load(global_feat_path).to(device)
        pass
    else:
        raise Exception(f"Encoder path {encoder_path} not exist")

    # if os.path.exist(decoder_path):
    #     # load
    #     decoder = torch.load(torch.load(decoder_path)).to(device)
    # else:
    #     raise Exception(f"Encoder path {decoder_path} not exist")

    graph_output_size=  encoder.output_size
    global_output_size = global_feat_extractor.output_size

    classifier = Classifier(
        graph_output_size = graph_output_size,
        global_output_size = global_output_size,
        hidden_size = args["classifier_param"]["hidden_size"],
        num_class = args["classifier_param"]["num_class"],
        dropout_rate = args["classifier_param"]["dropout_rate"]).to(device)
    if args["load_train"]:
        encoder.eval()
        global_feat_extractor.eval()
        list_param = list(classifier.parameters())
    else:
        encoder.train()
        global_feat_extractor.train()
        list_param = list(encoder.parameters())+list(global_feat_extractor.parameters())+list(classifier.parameters())
    return encoder, global_feat_extractor, classifier, list_param
'''
Encoder
Global Feature Extractor
TransformerDecoder/LSTM Decoder
trained already for the captioning tasks
Then we use it to train the classifier
'''

def model_save(args,classifier,best_eval_loss,best_epoch):
    base_save_path = args["model_save_base_path"]
    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path)

    eval_loss = f"{best_eval_loss:.2f}"

    classifier_name = args['save_classifier']+"-"+str(best_epoch)+"-"+str(eval_loss)
    classifier_store_path = os.path.join(base_save_path, classifier_name)
    torch.save(classifier, classifier_store_path)

def calc_acc_f1(gt_label,pred_label):
    accuracy = accuracy_score(pred_label, gt_label)
    f1 = f1_score(gt_label, pred_label, average='weighted') 
    return accuracy, f1

def eval(eval_loader, encoder,global_feature_extractor, classifier, device, criterion):
    batch_size = eval_loader.batch_size
    total_samples = len(eval_loader.dataset)
    total_step = math.ceil(total_samples / batch_size)
    total_loss = 0.0
    for i in tqdm(range(total_step)):
        cg, tg, assign_mat, caption_tokens, labels, captions, images = next(iter(eval_loader))
        cg = cg.to( device)
        tg = tg.to(device)
        images =  images.to(device)
        labels = labels.to(device)
        gt_labels = []
        pred_labels = []
        with torch.no_grad():
            out = encoder(cg,tg,assign_mat,images)
            global_feat = global_feature_extractor(images)
            merged_feat = torch.cat((out, global_feat), dim=1)
            pred_matrix = classifier(merged_feat)
            pred_matrix = pred_matrix.to(device)
            #print(f"EVAL pred matrix {pred_matrix.shape} labels shape {labels.shape}")
            loss = criterion(pred_matrix,labels)
            total_loss += loss
            pred_label = torch.argmax(pred_matrix, dim = 1)
            gt_labels.extend(labels.tolist())
            pred_labels.extend(pred_label.tolist())
        
        accuracy, f1_score = calc_acc_f1(gt_labels,pred_labels)
    return accuracy,f1_score, total_loss

def get_balance_dl(dataset,batch_size):
    class_count = {
    '0': 0,
    '1': 0,
    '2': 0,
    }

    count = 0
    class_weight = []
    for idx,output in enumerate(dataset):
        _, _, _, _, labels, _, _= output
        class_count[str(labels)] += 1
        count += 1

    for key,value in class_count.items():
        class_weight.append(1/value)
    print(class_weight)
    sample_weights = [0]*len(dataset)

    for idx, output in enumerate(dataset):
        _, _, _, _, labels, _, _= output
        # print(labels.shape)
        class_count[str(labels)] += 1
        sample_weights[idx] = class_weight[labels]
    sampler = WeightedRandomSampler(weights = sample_weights,num_samples = len(dataset),replacement = True)
    # print(list(sampler))

    dl = dataset_to_loader(dataset,batch_size = batch_size,sampler = sampler, num_workers = 0)
    return dl


def train_classifier():
    parser = argparser()
    args = parser.parse_args()
    args = update_argparser(args)
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if "new" in args["vocab_path"]:
        vocab_use = "new"
    else:
        vocab_use = "old"

    with open(args["vocab_path"], 'rb') as file:
        vocabs = pickle.load(file)

    vocab_size = len(vocabs)

    encoder, global_feature_extractor, classifier, list_param = get_all_models(args,device)
    #   make the dl here
    #   !!!!!!!!!!! Change it back to train
    train_dl,train_dataset = make_dataloader(
        batch_size = args["batch_size"],
        split = "train",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True,
    )

    # train_dl = get_balance_dl(train_dataset,args["batch_size"])

    test_dl,_ = make_dataloader(
        batch_size =1000, # there are 1000 set 1 because we will calculate pair by pair
        split = "test",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    eval_dl,_ = make_dataloader(
        batch_size = 889, # there are 889 in eval set
        split = "eval",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    train_dl = get_sample_samplier(train_dataset,args["batch_size"])
    

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    classifier_optimizer = torch.optim.Adam(params = list_param, lr = args["learning_rate"], weight_decay=args["weight_decay"])
    print(f"ready to init !!!!!")
    wandb.login(key = "ce5676f856caf561584c75f8175f6967876f1c77")
    wandb.init(
        project="GNN histopathology image captioning Classification",
        name = args["classifier_name"],
        #   track hyperparameters and run metadata
        config={
            "architecture": "Simple Classifier",
            "dataset": "Nmi-Wsi-Diagnosis",
            "epoch": args["epochs"],
            "batch_size": args["batch_size"],
            "num_param":len(list_param),
            "vocab_size":vocab_size,
            "loss":args["loss"],
            "encoder_use":args["load_encoder_name"],
            "global_feat_extractor":args["load_global_extractor_name"],
            "vocab_use":vocab_use,
        }
    )
    
    torch.autograd.set_detect_anomaly(True)

    total_samples = len(train_dl)
    print(f"lenght of tltal samples {total_samples}")
    batch_size = args["batch_size"]
    total_step = math.ceil(total_samples / batch_size)

    best_eval_loss = 10
    best_classifier = None
    best_epoch = 0
    for epoch in range(args["epochs"]):

        total_loss = []
        for step in range(total_step):
            cg, tg, assign_mat, caption_tokens, labels, caption, images = next(iter(train_dl))
            #print(f"caption tokens type{type(caption_tokens)} shape {caption_tokens.shape}")
            cg = cg.to(device)
            tg = tg.to(device)
            labels = labels.to(device)
            images = images.to(device)

            '''
            if we want to make encoder, global feat extractor trainable do it here
            '''

            out = encoder(cg,tg,assign_mat,images) # (batch_size, 1, embedding)
            global_feat = global_feature_extractor(images)

            merged_feat = torch.cat((out, global_feat), dim=1).unsqueeze(1)
            pred_matrix = classifier(merged_feat).to(device)

            '''
            calculate loss
            '''
            #print(f"pred matrix {pred_matrix.shape} labels shape {labels.shape}")
            class_loss = criterion(pred_matrix, labels)
            class_loss.backward()
            total_loss.append(class_loss.item())

            classifier_optimizer.zero_grad()
            classifier_optimizer.step()


        accuracy,f1_score,eval_loss = eval(eval_dl,encoder,global_feature_extractor, classifier, device, criterion)
        if eval_loss.item() < best_eval_loss:
            best_eval_loss = eval_loss.item()
            best_epoch = epoch
            best_classifier = classifier
        # if model_save:
        #     model_save(args,classifier,best_eval_loss, best_epoch)
        print(f"Epoch {epoch}: eval loss {eval_loss}")
        eval_output = {
            "train_loss":np.mean(total_loss),
            "eval_loss":eval_loss,
            "accuracy":accuracy,
            "f1_score":f1_score,
        }

        wandb.log(
            eval_output
        )
    test_accuracy,test_f1_score,_ = eval(train_dl,encoder,global_feature_extractor, classifier, device, criterion)
    print(f"FOR TRAIN test accuracy is {test_accuracy} test_f1_score {test_f1_score}")
    test_accuracy,test_f1_score,_ = eval(test_dl,encoder,global_feature_extractor, classifier, device, criterion)
    print(f"FOR TEST test accuracy is {test_accuracy} test_f1_score {test_f1_score}")



if __name__ == "__main__":
    train_classifier()

    # import os
    # encoder_path = "../../../../../../srv/scratch/bic/peter/model_save/Encoder2-99-2.95.pt"
    # if os.path.exists(encoder_path):
    #     # load
    #     print(f"found")
    #     pass
    # else:
    #     print(f"no")