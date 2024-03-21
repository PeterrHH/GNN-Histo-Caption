import os
from simple_parsing import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import math
from sklearn.metrics import f1_score, accuracy_score
import itertools
import wandb
import yaml
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from evaluation import Scorer
from dataloader import make_dataloader,dataset_to_loader
from models.CLIP import CLIPModel, AvgMeter
from models.Graph_Model import GNNEncoder
from models.LSTM2 import LSTMDecoder
from models.GlobalFeatureExtractor import GlobalFeatureExtractor

def model_def(args,device,vocabs):
    '''
    decoder: transformer/lstm
    '''
        #   Define Model, Loss and 
    encoder = GNNEncoder(
        args = args,
        cg_layer = args['gnn_param']['cell_layers'], 
        tg_layer = args['gnn_param']['tissue_layers'],
        aggregate_method = args['gnn_param']['aggregate_method'], 
        input_feat = 514,
        hidden_size = args['gnn_param']['hidden_size'],
        output_size = args['gnn_param']['output_size'],
    ).to(device)

    # attention = EncoderLayer(d_model = args['gnn_param']['output_size'], 
    #     nhead = 4, 
    #     dim_feedforward = 1024, 
    #     dropout = 0.2).to(device)


    decoder = LSTMDecoder(
        vocabs = vocabs, 
        #embed_size = args["global_class_param"]["output_size"], 
        embed_size = args['gnn_param']['output_size']+args["global_class_param"]["output_size"], 
        # embed_size = args["global_class_param"]["output_size"], 
        hidden_size = args["lstm_param"]["size"],  
        batch_size= args["batch_size"], 
        bi_direction = args["lstm_param"]["bi_direction"],
        device = device,
        dropout = args["lstm_param"]["dropout"],
        num_layers = args["lstm_param"]["num_layers"]
    ).to(device)


    global_feature_extractor = GlobalFeatureExtractor(
        hidden_size = args["global_class_param"]["hidden_size"],
        output_size = args["global_class_param"]["output_size"],
        dropout_rate = args["global_class_param"]["dropout_rate"]).to(device)


    return encoder, decoder, global_feature_extractor

def train_epoch(loader,model,batch_size,device, optimizer, lr_scheduler):
    loss_meter = AvgMeter()
    total_samples = len(loader)
    total_step = math.ceil(total_samples / batch_size)
    history = []
    for _ in range(total_step):
        cg, tg, assign_mat, caption_tokens, labels, caption, images,attention_mask = next(iter(loader))
        cg = cg.to(device)
        tg = tg.to(device)
        images = images.to(device)
        # assign_mat = assign_mat.to(DEVICE)
        caption_tokens = caption_tokens.to(device) 
        attention_mask = attention_mask.to(device)
        loss = model(cg,tg,assign_mat,images,caption_tokens,attention_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        loss_meter.update(loss.item(), batch_size)

        history.append(loss.cpu().detach().numpy().mean())
    return loss_meter 

def eval_epoch(loader,model,batch_size, device, optimizer, lr_scheduler):
    loss_meter = AvgMeter()

    total_samples = len(loader.dataset)
    total_step = math.ceil(total_samples / batch_size) 

    for _ in range(total_step):
        cg, tg, assign_mat, caption_tokens, labels, caption, images,attention_mask = next(iter(loader))
        cg = cg.to(device)
        tg = tg.to(device)
        images = images.to(device)
        # assign_mat = assign_mat.to(DEVICE)
        caption_tokens = caption_tokens.to(device) 
        attention_mask = attention_mask.to(device)

        loss = model(cg,tg,assign_mat,images,caption_tokens,attention_mask)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # lr_scheduler.step()

        loss_meter.update(loss.item(), batch_size)
    return loss_meter



def main():

    config_file_path =  "config/config.yaml"
    with open(config_file_path, "r") as conf_file:
        args = yaml.full_load(conf_file)

    print(args)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args["vocab_path"], 'rb') as file:
        vocabs = pickle.load(file)
    print(f"len vocab is {len(vocabs)}")
    wandb.init(
        project="GNN Contrastive Training",
        name = args["wandb_name"],
        #   track hyperparameters and run metadata
        config={
            "architecture": "GCN-LSTM",
            "dataset": "Nmi-Wsi-Diagnosis",
            "epoch": args["epochs"],
            "batch_size": args["batch_size"],
        }
    )

    train_dl,train_dataset = make_dataloader(
        batch_size = args["batch_size"],
        split = "train",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    # #train_dl = get_sample_samplier(train_dataset,args["batch_size"])
    # print(f"train loader size {len(train_dl)}")

    test_dl,_ = make_dataloader(
        batch_size =args["batch_size"], # there are 1000 set 1 because we will calculate pair by pair
        split = "test",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    eval_dl,_ = make_dataloader(
        batch_size = args["batch_size"], # there are 889 in eval set
        split = "eval",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    model = CLIPModel(args = args,device=DEVICE).to(DEVICE)
    batch_size = args["batch_size"]
    params = [
        {"params": model.graph_encoder.parameters(), "lr": 1e-4},
        {"params":model.feature_extractor.parameters(),"lr":1e-4},
        {"params": model.text_encoder.parameters(), "lr": 1e-5},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": 1e-3, "weight_decay": 1e-3}
    ]
    optimizer = torch.optim.Adam(params = params,  weight_decay=args["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=1, factor=0.8
    )

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    best_loss = float('inf')
    for epoch in range(args["epochs"]):
        model.train()
        train_loss = train_epoch(train_dl,model,batch_size,DEVICE,optimizer,lr_scheduler)
        print(f"train loss for epochs {epoch} is {train_loss} type is {type(train_loss)}")
        model.eval()
        with torch.no_grad():
            valid_loss = eval_epoch(eval_dl,model,batch_size,DEVICE,optimizer,lr_scheduler)
        print(f"train loss for epochs {epoch} is {train_loss} type is {train_loss.avg} valid loss is {valid_loss}")
        wandb.log({'train_loss':train_loss.avg,'valid_loss':valid_loss.avg})

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model, f"../../../../../../srv/scratch/bic/peter/CLIP_save/best_{batch_size}.pt")
            print("Saved Best Model!")
        lr_scheduler.step(valid_loss.avg)

if __name__ == "__main__":
    main()