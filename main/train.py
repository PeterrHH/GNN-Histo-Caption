import os
from simple_parsing import ArgumentParser
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import math

import yaml
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from evaluation import Scorer
from dataloader import make_dataloader

import wandb

from argparse import ArgumentParser
from models.BaseModel import GNN_LSTM
from Vocabulary import Vocabulary
from models.Graph_Model import GNNEncoder
from models.LSTM import LSTMDecoder
from data_plotting import violin_plot



# parser = ArgumentParser()
# parser.add_argument('echo', help = 'echo the given string')
# parser.add_argument('-n','--number',help = "number", type = int, default = 0, nargs = '?')
# parser.add_argument('-v','--verbose',help = "Provide disc", action = "store_true") # if --verbose has value in command line, then, it is true, 
# parser.add_argument('-w','--weight',help = "Wegihts", type = int, choices = [0,1,2]) # 

# args = parser.parse_args()


# if args.verbose:
#     print(args.echo)
# else:
#     print("None verbose")
    
# if args.weight == 0:
#     print("Weight is 0")
# elif args.weight == 1:
#     print("Weight is 1")
# else:
#     print(f"{args.weight}")
import sys
sys.path.append('../histocartography/histocartography')
from preprocessing import (
    VahadaneStainNormalizer,         # stain normalizer
    NucleiExtractor,                 # nuclei detector 
    DeepFeatureExtractor,            # feature extractor 
    KNNGraphBuilder,                 # kNN graph builder
    ColorMergedSuperpixelExtractor,  # tissue detector
    DeepFeatureExtractor,            # feature extractor
    RAGGraphBuilder,                 # build graph
    AssignmnentMatrixBuilder         # assignment matrix 
)

print("OK")

''' 
Update relevant arguments
input: args read and the content of config_file
'''
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
For a batch
'''
def embed2sentence(decode_output, loader, captions, pred_dict, cap_dict):
    # print(f"Decode output is {decode_output}")
   # print(f"Decode_output is {decode_output.shape}")
    #print(f"Captions here is {len(captions)}")
    # max_indices = torch.argmax(decode_output, dim=2) 

    # assert len(pred_dict) == len(cap_dict)


    for i,caption in enumerate(captions):
        for sent_i,sent_cap in enumerate(caption):
            captions[i][sent_i] = ' '.join(sent_cap.split()).replace("<pad>", "").replace("<end>", ".").replace("<start>","")
    j = 0
    for idx,embed in enumerate(decode_output):

        sentence = " ".join([loader.dataset.vocab.idx2word[int(idx)] for idx in embed])
        sentence = sentence.replace("<pad>","").replace("<start>","")
        sentence = ' '.join(sentence.split()).replace("<end>", ".")
        if len(pred_dict.keys()) == 0:
            #   Empty
            pred_dict["1"] = [sentence]
            cap_dict["1"] = captions[idx]
            print(f"-------Prediction------")
            print(pred_dict)
            print(f"---------cap dict-------")
            print(cap_dict)
            pass
        else:
            pred_dict[str(len(pred_dict)+1)] = [sentence]
            cap_dict[str(len(cap_dict)+1)] = captions[idx]
    
    print(f"length pred {len(pred_dict)} and leng cap {len(cap_dict)}")
    # print(f"EXP-------------")
    # print(pred_dict)
    # print(cap_dict)
    return pred_dict,cap_dict


def eval(eval_loader,encoder,decoder,device, batch_size,criterion, vocab_size, eval = True) :
    total_samples = len(eval_loader.dataset)
    total_step = math.ceil(total_samples / batch_size)
    pred_dict = {}
    cap_dict = {}
    test_output = {
        "Bleu1":[],
        "Bleu2":[],
        "Bleu3":[],
        "Bleu4":[],
        "METEOR":[],
        "ROUGE_L":[],
        "CIDEr":[],
        "SPICE":[]
    }
    print(f"total sample is {total_samples} total step is {total_step} batch_size is {batch_size}")
    print(f"TOTAL STEP is {total_step}")
    for step in tqdm(range(total_step)):
        cg, tg, assign_mat, caption_tokens, labels, captions = next(iter(eval_loader))
        # caption_dict = {str(i + 1): value for i, value in enumerate(captions)}
        #print(f"Length of labels {labels.shape}")
        cg = cg.to( device)
        tg = tg.to(device)
        print(f"cell graph {cg} tissue graph {tg} assign mat len {len(assign_mat)} shape {assign_mat[0].shape}")

        encoder , decoder = encoder.to(device) , decoder.to(device)
        caption_tokens = caption_tokens.to(device)
        # encoder.eval()
        # decoder.eval()
        with torch.no_grad():
            out = encoder(cg,tg,assign_mat)
            lstm_out, lstm_out_tensor = decoder.predict(out,90)
            #print("LSTM OUT HERE")
        #   Evaluate
        if eval:
        # print(f"In eval lstm_out {lstm_out.shape} and cap token is {caption_tokens.shape}")
            eval_loss = criterion(lstm_out_tensor.view(-1, vocab_size) , caption_tokens.view(-1) )
        pred_dict,cap_dict = embed2sentence(lstm_out,eval_loader,captions,pred_dict,cap_dict)
        # print(f"---------------------------------------")
        # print(cap_dict)
        # print(f"-------------CAP DICT ABOVE------------")
        # print(pred_dict)
        # print(f"-------------Pred DICT ABOVE-------------")
        # scorer = Scorer(cap_dict,pred_dict)
        scorer = Scorer(cap_dict,pred_dict)
        # if eval:
        scores = scorer.compute_scores()
        if eval is False:
            for key,value in scores.items():
                test_output[key].append(value[0])
        # else:
        #     scores = scorer.compute_scores_iterative()
        # packed = unpack_score(scores)
        # print(packed)
        # for i, key in enumerate(test_output.keys()):

        #     test_output[key].append(packed[i])
    if eval:
        # compute only mean
        test_output = {key: value[0] for key, value in scores.items()}
    
    else:
        # compute mean and std
        print(f"------the scores in test: below ---------")
        '''
        Scores example
        {'Bleu1': [0.24373861938874267], 
         'Bleu2': [0.14339670728907034], 
         'Bleu3': [0.09570660997081161], 
         'Bleu4': [0.06831824715884934], 
         'METEOR': [0.17968662760237566], 
         'ROUGE_L': [0.2821534687597159], 
         'CIDEr': [2.6368785118009425e-09], 
         'SPICE': [0.2266488810373783]}
        '''
        print(scores)
        #violin_plot(scores,"GCN-LSTM-40eps")
        for key, values in test_output.items():
            print(f'length of values in {len(values)}')
            mean_value = np.mean(values)  # Calculate the mean using np.mean
            std_value = np.std(values)  
            test_output[key] = [mean_value,std_value] # Store the mean in the new dictionary
        print(f"In test")
        print(test_output)
    return test_output,eval_loss

def save_model_to_path(epoch, encoder, encoder_path,decoder, decoder_path, encoder_name,decoder_name):
    if not os.path.exists(encoder_path):
        os.mkdir(encoder_path)
    if not os.path.exists(decoder_path):
        os.mkdir(decoder_path) 

    encoder_store_path = os.path.join(encoder_path, encoder_name)
    decoder_store_path = os.path.join(decoder_path, decoder_name)

    torch.save(encoder, encoder_store_path)
    torch.save(decoder, decoder_store_path)
    pass
def load_model(encoder_path, decoder_path,vocab_size,args,device):
    encoder = GNNEncoder(
        args = args,
        pool_method = None, 
        cg_layer = args['gnn_param']['cell_layers'], 
        tg_layer = args['gnn_param']['tissue_layers'],
        aggregate_method = "sum", 
        input_feat = 512,
        output_size = 256
    )
    decoder = LSTMDecoder(
        vocab_size = vocab_size, 
        embed_size = 256, 
        hidden_size = 128,  
        batch_size= args["batch_size"], 
        device = device
    )
    if os.path.exist(encoder_path):
        # load
        encoder.load(torch.load(encoder_path))
        pass
    else:
        raise Exception(f"Encoder path {encoder_path} not exist")
    
    
    if os.path.exist(decoder_path):
        # load
        decoder.load(torch.load(decoder_path))
    else:
        raise Exception(f"Encoder path {decoder_path} not exist")
    
    return encoder, decoder

def unpack_score(scores):
    bleu1 = scores["Bleu1"]
    bleu2 = scores["Bleu2"]
    bleu3 = scores["Bleu3"]
    bleu4 = scores["Bleu4"]
    meteor = scores["METEOR"]
    rouge = scores["ROUGE_L"]
    cider = scores["CIDEr"]
   # spice = scores["SPICE"]

    return bleu1, bleu2, bleu3, bleu4, meteor, rouge, cider


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False

    #   Get all the parser
    parser = argparser()
    args = parser.parse_args()
    print(args)
    print("Parse")
    args = update_argparser(args)
    print(args)
    learning_rate = args["learning_rate"]
    gnn_param = args["gnn_param"]
    lstm_param = args["lstm_param"]
    #   set path to save checkpoints

    #   make the dl here
    #   !!!!!!!!!!! Change it back to train
    train_dl = make_dataloader(
        batch_size = args["batch_size"],
        split = "train",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    test_dl = make_dataloader(
        batch_size = args["batch_size"], # there are 1000 set 1 because we will calculate pair by pair
        split = "test",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    eval_dl = make_dataloader(
        batch_size = 889, # there are 889 in test set
        split = "eval",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    #   Define Model, Loss and 
    vocab_size = len(train_dl.dataset.vocab)
    encoder = GNNEncoder(
        args = args,
        cg_layer = args['gnn_param']['cell_layers'], 
        tg_layer = args['gnn_param']['tissue_layers'],
        aggregate_method = "sum", 
        input_feat = 514,
        output_size = 128
    )


    decoder = LSTMDecoder(
        vocab_size = vocab_size, 
        embed_size = 128, 
        hidden_size = 128,  
        batch_size= args["batch_size"], 
        device = DEVICE
    )
    encoder , decoder = encoder.to(DEVICE) , decoder.to(DEVICE)
 
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    all_params = list(decoder.parameters())  + list( encoder.parameters() )
    optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"], weight_decay=args["weight_decay"])
    
    MAX_LENGTH = 100
    # model = GNN_LSTM(encoder, decoder,hidden_dim, vocab_size,gnn_param, lstm_param, phase)
    # model.to(DEVICE)
    torch.autograd.set_detect_anomaly(True)
    if args["phase"] == "train":
        #   Model training

        #   set the wandb project where this run will be logged
        wandb.init(
            project="GNN simplifcation and application in histopathology image captioning",
            name = "GSage-LSTM mean-agg bs32 with relu (3 CL 1 TL) wd = 0.0001",
            #   track hyperparameters and run metadata
            config={
                "architecture": "GCN-LSTM",
                "dataset": "Nmi-Wsi-Diagnosis",
                "epoch": args["epochs"],
                "batch_size": args["batch_size"]
            }
        )
        total_samples = len(train_dl)
        batch_size = 4
        total_step = math.ceil(total_samples / batch_size)
        
        print(f"Number of steps per epoch: {total_step}")
        print(type(train_dl))
        for epoch in range(args["epochs"]):
            # encoder.train()
            # decoder.train()
            total_loss = []
            # for batched_idx, batch_data in enumerate(tqdm(train_dl)):
            for step in range(total_step):
                #if args["graph_model_type"] == "Hierarchical":
                cg, tg, assign_mat, caption_tokens, labels, caption = next(iter(train_dl))

                cg = cg.to(DEVICE)
                tg = tg.to(DEVICE)
                # assign_mat = assign_mat.to(DEVICE)
                caption_tokens = caption_tokens.to(DEVICE) # (batch_size, num_sentences, num_words_in_sentence) num_sentence = 6, num_words = 16
                # print(encoder)

                encoder.zero_grad()    
                decoder.zero_grad()
                # print(f"Input shape is {cg}")
                out = encoder(cg,tg,assign_mat) # (batch_size, 1, embedding)
                print(f"encoder out shape is {out.shape} caption token shape {caption_tokens.shape}")
                lstm_out = decoder(out,caption_tokens)
               # print(f"caption shape {caption_tokens.shape} lstm shape is {lstm_out.shape}")
                #At the end, run eval set  
                lstm_out_prep = lstm_out.view(-1, vocab_size)
                caption_prep = caption_tokens.view(-1) 
                # print(f"before shape {lstm_out.shape} cap token {caption_tokens.shape}")
                # print(f"first ist {lstm_out.view(-1, vocab_size).shape} cap token {caption_tokens.view(-1).shape}")
                loss = criterion(lstm_out.view(-1, vocab_size) , caption_tokens.view(-1) )

                #loss = criterion(lstm_out.view(-1, vocab_size) , caption_tokens)
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                total_loss.append(loss.item())
                #print(f"FEAT SIZE IS {cg.ndata['feat'].shape}")
                # print(cg.ndata['feat'])
            mean_loss = np.mean(total_loss)
            # wandb.log({'trian_loss':loss})
            del total_loss
            print(f"Eval set evaluating mean loss as {mean_loss} in epoch {epoch}")
            scores,eval_loss = eval(eval_dl,encoder,decoder,DEVICE,889,criterion,vocab_size)
            # bleu1, bleu2, bleu3, bleu4, meteor, rouge, cider, spice = unpack_score(scores) # mean and standard dev
            eval_output = {
                'train_loss':mean_loss,
                'eval_loss':eval_loss,
                'bleu1':scores['Bleu1'],
                'bleu2':scores['Bleu2'],
                'bleu3':scores['Bleu3'],
                'bleu4':scores['Bleu4'],
                'meteor':scores['METEOR'],
                'rouge':scores['ROUGE_L'],
                'cider':scores['CIDEr'],
                #'spice':scores['SPICE'],
            }
            wandb.log(eval_output)
            print(f"!!!At epoch [{str(epoch+1)}/{args['epochs']}] evaluate results is {eval_output}")
            if args["save_model"]:
                if (epoch+1) % args["save_every"] == 0:
                    encoder_name = f"Gsage_encoder_epoch{str(epoch+1)}.pt"
                    decoder_name = f"Gsage_LSTM_decoder_epoch{str(epoch+1)}.pt"
                    save_model_to_path(epoch, encoder, args["encoder_path"],decoder, args["decoder_path"], encoder_name,decoder_name)

            torch.cuda.empty_cache()

        scores,_ = eval(test_dl,encoder,decoder, DEVICE, args["batch_size"],criterion,vocab_size,eval = False)

        test_output = {
                'bleu1_mean':scores['Bleu1'][0],
                'bleu1_std':scores['Bleu1'][1],
                'bleu2_mean':scores['Bleu2'][0],
                'bleu2_std':scores['Bleu2'][1],
                'bleu3_mean':scores['Bleu3'][0],
                'bleu3_std':scores['Bleu3'][1],
                'bleu4_mean':scores['Bleu4'][0],
                'bleu4_std':scores['Bleu4'][1],
                'meteor_mean':scores['METEOR'][0],
                'meteor_std':scores['METEOR'][1],
                'rouge_mean':scores['ROUGE_L'][0],
                'rouge_std':scores['ROUGE_L'][1],
                'cider_mean':scores['CIDEr'][0],
                'cider_std':scores['CIDEr'][1],
                #'spice_mean':scores['SPICE'][0],
                #'spice_std':scores['SPICE'][1],
            }
        print("Testing data output: ")
        print(test_output)


    else:
        #   Only run the Test set
        print("test")



if __name__ == "__main__":
    main()
    # vocab = Vocabulary()
    # vocab.print_all_words()


