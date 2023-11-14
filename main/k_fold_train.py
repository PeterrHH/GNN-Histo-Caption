import os
from simple_parsing import ArgumentParser
import random
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
import multiprocessing
import numpy as np
import pandas as pd
import torch
from IPython.utils import io
import torch.nn as nn
import pickle
from tqdm import tqdm
import math

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

    #print(captions)
    for i,caption in enumerate(captions):
        for sent_i,sent_cap in enumerate(caption):
        #    print(captions[i][sent_i])
           captions[i][sent_i] = ' '.join(sent_cap.split()).replace("<pad>", "").replace("<end>", ".").replace("<start>","")
        #captions[i] = ' '.join(caption.split()).replace("<pad>", "").replace("<end>", ".").replace("<start>","")

    j = 0
    for idx,embed in enumerate(decode_output):

        sentence = " ".join([loader.dataset.vocab.idx2word[int(idx)] for idx in embed])
        sentence = sentence.replace("<pad>","").replace("<start>","")
        sentence = ' '.join(sentence.split()).replace("<end>", ".")
        if len(pred_dict.keys()) == 0:
            #   Empty
            pred_dict["1"] = [sentence]
            cap_dict["1"] = captions[idx]
            pass
        else:
            pred_dict[str(len(pred_dict)+1)] = [sentence]
            cap_dict[str(len(cap_dict)+1)] = captions[idx]
    
   # print(f"length pred {len(pred_dict)} and leng cap {len(cap_dict)}")
    # print(f"E----cap dict-------")
    # print(cap_dict)
    return pred_dict,cap_dict


def eval(eval_loader,encoder,attention,global_feature_extractor, decoder,device, batch_size,criterion, vocab_size, eval = True) :
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
    #print(f"total sample is {total_samples} total step is {total_step} batch_size is {batch_size}")
    #print(f"TOTAL STEP is {total_step}")
    for step in tqdm(range(total_step)):
        cg, tg, assign_mat, caption_tokens, labels, captions, images = next(iter(eval_loader))
        # caption_dict = {str(i + 1): value for i, value in enumerate(captions)}
        #print(f"Length of labels {labels.shape}")
        cg = cg.to( device)
        tg = tg.to(device)
        images = images.to(device)
 
        # print(f"caption_tokens shape {caption_tokens.shape}")
        # print(f"caption len{len(captions)} within {len(captions[0])}")
        encoder , decoder, attention = encoder.to(device) , decoder.to(device), attention.to(device)
        caption_tokens = caption_tokens.to(device)
        # encoder.eval()
        # decoder.eval()
        with torch.no_grad():
            out = encoder(cg,tg,assign_mat,images)
            global_feat = global_feature_extractor(images)
            merged_feat =  torch.cat((out, global_feat), dim=1)
            # out = attention(out)
            lstm_out, lstm_out_tensor = decoder.predict(merged_feat,90)
            #print(f"lstm out {lstm_out.shape} and tensor {lstm_out_tensor.shape}")
            #print("LSTM OUT HERE")
        #   Evaluate
        if eval:
            # print(f"In eval lstm_out {lstm_out.shape} and cap token is {caption_tokens.shape}")
           # print(f"lstm view shape {lstm_out_tensor.view(-1, vocab_size).shape} and cap view {caption_tokens.view(-1).shape}")
            #eval_loss = criterion(lstm_out_tensor.view(-1, vocab_size) , caption_tokens.view(-1) )
            all_eval_loss = []
            for cap_idx in range(caption_tokens.size(1)):
                all_eval_loss.append(criterion(lstm_out_tensor.view(-1, vocab_size),caption_tokens[:,cap_idx,:].reshape(-1)))
            eval_loss = sum(all_eval_loss) / len(all_eval_loss)
        else:
            eval_loss = None

        #eval_loss = criterion(lstm_out_tensor.view(-1, vocab_size) , caption_tokens.view(-1) )
        pred_dict,cap_dict = embed2sentence(lstm_out,eval_loader,captions,pred_dict,cap_dict)
        # print(f"-------cap_dict-------")
        # print(cap_dict['1'])
        # print(f'-----pred_dict------')
        # print(pred_dict['1'])
        # print(f"cap dict len{len(cap_dict)} pred dict {len(pred_dict)}")


        scorer = Scorer(cap_dict,pred_dict)
        # if eval:
        with io.capture_output() as captured:
            scores = scorer.compute_scores()
        if eval is False:
            for key,value in scores.items():
                test_output[key].append(value[0])

    if eval:
        # compute only mean
        test_output = {key: value[0] for key, value in scores.items()}
    
    else:
        # compute mean and std
        #print(f"------the scores in test: below ---------")
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
        #print(scores)
        #violin_plot(scores,"GCN-LSTM-40eps")
        for key, values in test_output.items():
            # print(f'length of values in {len(values)}')
            mean_value = np.mean(values)  # Calculate the mean using np.mean
            std_value = np.std(values)  
            test_output[key] = [mean_value,std_value] # Store the mean in the new dictionary
        # print(f"In test")
        # print(test_output)
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
        output_size = 512
    )
    decoder = LSTMDecoder(
        vocab_size = vocab_size, 
        embed_size = 512, 
        hidden_size = 512,  
        batch_size= args["batch_size"], 
        bi_direction = args["lstm_param"]["bi_direction"],
        device = device,
        dropout = args["lstm_param"]["dropout"],
        num_layers = args["lstm_param"]["num_layers"]
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

def model_def(args,device,vocab_size,decoder_type = "transformer"):
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

    attention = EncoderLayer(d_model = args['gnn_param']['output_size'], 
        nhead = 4, 
        dim_feedforward = 1024, 
        dropout = 0.2).to(device)

    
    if decoder_type == "Transformer":
        decoder =  TransformerDecoder(
            vocab_size = vocab_size,
            d_model =  args['gnn_param']['output_size']+args["global_class_param"]["output_size"],
            nhead = args['transformer_param']['n_head'], 
            num_layers = args['transformer_param']['num_layers'], 
            dim_feedforward=args['transformer_param']['dim_feedforward'], 
            dropout= args['transformer_param']['dropout'],
        ).to(device)
    elif decoder_type == "LSTM":
        decoder = LSTMDecoder(
            vocab_size = vocab_size, 
            embed_size = args['gnn_param']['output_size']+args["global_class_param"]["output_size"], 
            hidden_size = args["lstm_param"]["size"],  
            batch_size= args["batch_size"], 
            bi_direction = args["lstm_param"]["bi_direction"],
            device = device,
            dropout = args["lstm_param"]["dropout"],
            num_layers = args["lstm_param"]["num_layers"]
        ).to(device)
    '''

    decoder =  TransformerDecoder(
        vocab_size = vocab_size,
        d_model = args['gnn_param']['output_size'],
        nhead = 4, 
        num_layers = 3, 
        dim_feedforward=2048, 
        dropout=0.1
    )
    '''
    global_feature_extractor = GlobalFeatureExtractor(
        hidden_size = args["global_class_param"]["hidden_size"],
        output_size = args["global_class_param"]["output_size"],
        dropout_rate = args["global_class_param"]["dropout_rate"]).to(device)

    classifier = Classifier(
        graph_output_size = args['gnn_param']['output_size'],
        global_output_size = args["global_class_param"]["output_size"],
        hidden_size = args["classifier_param"]["hidden_size"],
        num_class = args["classifier_param"]["num_class"],
        dropout_rate = args["classifier_param"]["dropout_rate"]).to(device)
    return encoder, attention, decoder, global_feature_extractor, classifier 

def train_epoch(loader,batch_size,device,encoder,decoder,attention,global_feature_extractor,criterion,all_params,optimizer,vocab_size):

    total_samples = len(loader)
    batch_size = batch_size
    total_step = math.ceil(total_samples / batch_size)
    total_loss = []
    for step in range(total_step):
        #if args["graph_model_type"] == "Hierarchical":
        cg, tg, assign_mat, caption_tokens, labels, captions, images = next(iter(loader))

        cg = cg.to(device)
        tg = tg.to(device)
        caption_tokens = caption_tokens.to(device) # (batch_size, num_sentences, num_words_in_sentence) num_sentence = 6, num_words = 16
        # print(encoder)
        images = images.to(device)


        encoder.zero_grad()   
        global_feature_extractor.zero_grad() 
        decoder.zero_grad()
        # print(f"Input shape is {cg}")
        out = encoder(cg,tg,assign_mat,images) # (batch_size, 1, embedding)
        # out = attention(out)
        global_feat = global_feature_extractor(images)
        merged_feat = torch.cat((out, global_feat), dim=1)
        out = decoder(merged_feat,caption_tokens)

        loss = criterion(out.view(-1, vocab_size) , caption_tokens.view(-1) )
        
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(all_params, 5.0)
        optimizer.step()
        total_loss.append(loss.item())
        #loss = criterion(lstm_out.view(-1, vocab_size) , caption_tokens)
    return encoder,decoder,attention,total_loss



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

    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #   make the dl here
    #   !!!!!!!!!!! Change it back to train
    train_dl, train_dataset = make_dataloader(
        batch_size = args["batch_size"],
        split = "train",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    test_dl, test_dataset = make_dataloader(
        batch_size =1000, # there are 1000 set 1 because we will calculate pair by pair
        split = "test",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )

    eval_dl, eval_dataset = make_dataloader(
        batch_size = 889, # there are 889 in eval set
        split = "eval",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    '''
    dataset = ConcatDataset([train_dataset, eval_dataset])
    '''
    dataset = train_dataset
    #   Define Model, Loss and 
    vocab_size = len(train_dl.dataset.vocab)


 






    # model = GNN_LSTM(encoder, decoder,hidden_dim, vocab_size,gnn_param, lstm_param, phase)
    # model.to(DEVICE)

    if args["phase"] == "train":
        #   Model training

        #   set the wandb project where this run will be logged
        wandb.init(
            project="GNN simplifcation and application in histopathology image captioning",
            name = args["wandb_name"],
            #   track hyperparameters and run metadata
            config={
                "architecture": "GCN-LSTM",
                "dataset": "Nmi-Wsi-Diagnosis",
                "epoch": args["epochs"],
                "batch_size": args["batch_size"],
                # "num_param":len(all_params),
            }
        )
        total_samples = len(train_dl)
        batch_size = args["batch_size"]
        total_step = math.ceil(total_samples / batch_size)
        


        kfold = KFold(n_splits=5, shuffle=True, random_state=42) 

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"FOLD: {fold}")
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)

            train_loader = dataset_to_loader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = dataset_to_loader(dataset, batch_size=total_samples, sampler=test_sampler)

            encoder, attention, decoder, global_feature_extractor, classifier = model_def(args, DEVICE, vocab_size,decoder_type="LSTM")
            criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
            all_params = list(decoder.parameters())  + list( encoder.parameters() ) + list(global_feature_extractor.parameters())

            if args["optimizer_type"] == "Adam":
                optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"], weight_decay=args["weight_decay"])
                #optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"])
            elif args["optimizer_type"] == "SGD":
                optimizer = torch.optim.SGD(params=all_params, lr=args["learning_rate"],weight_decay=args["weight_decay"])
            torch.autograd.set_detect_anomaly(True)

            encoder , decoder, attention = encoder.to(DEVICE) , decoder.to(DEVICE), attention.to(DEVICE)
            for epoch in range(args["epochs"]):
                encoder,decoder,attention,total_loss = train_epoch(train_loader,args["batch_size"],DEVICE,encoder,decoder,attention,global_feature_extractor,criterion,all_params,optimizer,vocab_size)
                mean_loss = np.mean(total_loss)
                print(f"Epoch {str(epoch+1)}: train loss is {mean_loss}")

                del total_loss
            scores,eval_loss = eval(eval_dl,encoder,attention,global_feature_extractor,decoder,DEVICE,889,criterion,vocab_size)

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

        scores,_ = eval(test_dl,encoder,attention,decoder, DEVICE, args["batch_size"],criterion,vocab_size,eval = False)

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
    # multiprocessing.set_start_method('spawn', force=True)
    main()
