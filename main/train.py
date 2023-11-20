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
from sklearn.metrics import f1_score, accuracy_score

import yaml
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from evaluation import Scorer
from dataloader import make_dataloader

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


    # print(f"---embed2sentence")
    # print(decode_output[0])
    # print(captions[0])
    

    for i,caption in enumerate(captions):
        for sent_i,sent_cap in enumerate(caption):
            captions[i][sent_i] = ' '.join(sent_cap.split()).replace("<pad>", "").replace("<end>", ".").replace("<start>","")
            
    j = 0
    for idx,embed in enumerate(decode_output):

        sentence = " ".join([loader.dataset.vocab.idx2word[int(idx)] for idx in embed])
        sentence = sentence.replace("<pad>","").replace("<start>","")
        sentence = ' '.join(sentence.split()).replace("<end>", ".")
        #print(f"length pred_dict {len(pred_dict.keys())} and length cap {len(cap_dict.keys())}")
        if len(pred_dict.keys()) == 0:
            #   Empty
            pred_dict["1"] = [sentence]
            # for i in range(5):
            #     cap_dict[str(int(i)+1)] = captions[idx]
            cap_dict["1"] = captions[idx]
            print(f"-------Prediction------")
            print(pred_dict)
            print(f"---------cap dict-------")
            print(cap_dict)
            pass
        else:
            pred_dict[str(len(pred_dict)+1)] = [sentence]
            cap_dict[str(len(cap_dict)+1)] = captions[idx]
    
   # print(f"length pred {len(pred_dict)} and leng cap {len(cap_dict)}")
    # print(f"E----cap dict-------")
    # print(cap_dict)
    return pred_dict,cap_dict

def calc_acc_f1(gt_label,pred_matrix,batch_size,device):
    pred_label = torch.argmax(pred_matrix, dim=1)
    #print(f"shape of pred_matrix is {pred_matrix.shape} pred label {pred_label.shape} gt_label {gt_label.shape} ")
    #print(f"------First label {pred_label} and gt {gt_label[0]}------")

    # correct_predictions = (pred_label == gt_label).sum().item()
    # accuracy = correct_predictions / batch_size
    accuracy = accuracy_score(pred_label.cpu().numpy(), gt_label.cpu().numpy())
    f1 = f1_score(gt_label.cpu().numpy(), pred_label.cpu().numpy(), average='weighted') 
    return accuracy, f1


def eval(eval_loader,
         encoder,attention,decoder, global_feature_extractor, classifier,
         device,
         criterion, vocab_size, labels,eval = True) :
    batch_size = eval_loader.batch_size
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
        # "SPICE":[]
    }
    #print(f"total sample is {total_samples} total step is {total_step} batch_size is {batch_size}")
    #print(f"TOTAL STEP is {total_step}")
    for step in tqdm(range(total_step)):
        cg, tg, assign_mat, caption_tokens, labels, captions, images = next(iter(eval_loader))
        # caption_dict = {str(i + 1): value for i, value in enumerate(captions)}
        # print(f"Length of labels {labels.shape}")
        cg = cg.to( device)
        tg = tg.to(device)
        images =  images.to(device)
        labels = labels.to(device)
        # encoder , decoder, attention = encoder.to(device) , decoder.to(device), attention.to(device)
        caption_tokens = caption_tokens.to(device)
        with torch.no_grad():
            out = encoder(cg,tg,assign_mat,images)
            global_feat = global_feature_extractor(images)
            #merged_feat = torch.cat((out, global_feat), dim=1)
            merged_feat = global_feat
            lstm_out, lstm_out_tensor = decoder.predict(merged_feat,90)
            pred_matrix = classifier(merged_feat)
            pred_matrix = pred_matrix.to(device)
        #   Evaluate
        if eval:
            print(f"In eval lstm_out {lstm_out.shape} and cap token is {caption_tokens.shape}")
           # print(f"lstm view shape {lstm_out_tensor.view(-1, vocab_size).shape} and cap view {caption_tokens.view(-1).shape}")
            #eval_loss = criterion(lstm_out_tensor.view(-1, vocab_size) , caption_tokens.view(-1) )        
            all_eval_loss = []
            for cap_idx in range(caption_tokens.size(1)):
                eval_cap_loss = criterion(lstm_out_tensor.view(-1, vocab_size),caption_tokens[:,cap_idx,:].reshape(-1))
                # eval_label_loss = criterion(pred_matrix,labels)
                # eval_loss = eval_cap_loss+ 0.5*eval_label_loss
                #eval_loss = eval_label_loss
                all_eval_loss.append(eval_cap_loss)
            eval_loss = sum(all_eval_loss) / len(all_eval_loss)
        else:
            eval_loss = None
        print(f"IN EVAL !!!!")
    
        accuracy,f1_score = calc_acc_f1(labels,pred_matrix,batch_size,device)
        #eval_loss = criterion(lstm_out_tensor.view(-1, vocab_size) , caption_tokens.view(-1) )
        pred_dict,cap_dict = embed2sentence(lstm_out,eval_loader,captions,pred_dict,cap_dict)
        # print(f"-------cap_dict-------")
        # print(cap_dict)
        # print(f'-----pred_dict------')
        # print(pred_dict)
        print(f"cap dict len{len(cap_dict)} pred dict {len(pred_dict)}")
        scorer = Scorer(cap_dict,pred_dict)
        # if eval:
        scores = scorer.compute_scores()
        if eval is False:
            for key,value in scores.items():
                test_output[key].append(value[0])

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
    return test_output,eval_loss,accuracy,f1_score

def save_model_to_path(args,encoder,decoder,global_feature_extractor,best_eval_loss,best_epoch):
    base_save_path = args["model_save_base_path"]
    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path)
    eval_loss = f"{best_eval_loss:.2f}"
    if encoder:
        print("args['save_encoder']")
        print(str(best_epoch))
        print(str(eval_loss))
        print()
        encoder_name = args['save_encoder']+"-"+str(best_epoch)+"-"+str(eval_loss)+".pt"
        encoder_store_path = os.path.join(base_save_path, encoder_name)
        torch.save(encoder, encoder_store_path)
        print(f"Encoder {encoder_name} SAVED")
    if global_feature_extractor:
        global_extractor_name = args['save_global_feature_extractor']+"-"+str(best_epoch)+"-"+str(eval_loss)+".pt"
        global_extractor_path = os.path.join(base_save_path,global_extractor_name)
        torch.save(global_feature_extractor,global_extractor_path)
        print(f"Global Extractor {global_extractor_name} SAVED")

    if decoder:
        decoder_name = args['save_decoder']+"-"+str(best_epoch)+"-"+str(eval_loss)+".pt"
        decoder_store_path = os.path.join(base_save_path, decoder_name)
        torch.save(decoder, decoder_store_path)
        print(f"Decoder {decoder_name} SAVED")


def save_classifier_to_path(args,classifier):
    base_save_path = args["model_save_base_path"]
    if not os.path.exists(base_save_path):
        os.mkdir(base_save_path)

    classifier_path = os.path.join(base_save_path, args['save_classifier'])

    pass
def load_model(encoder_path, decoder_path,vocabs,args,device):
    encoder = GNNEncoder(
        args = args,
        pool_method = None, 
        cg_layer = args['gnn_param']['cell_layers'], 
        tg_layer = args['gnn_param']['tissue_layers'],
        aggregate_method = "sum", 
        input_feat = 512,
        output_size = 512
    )
    '''
    
    '''
    decoder = LSTMDecoder(
        vocab_size = vocabs, 
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

'''
decoder transformer/LSTM
'''
def model_def(args,device,vocabs,decoder_type = "transformer"):
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
    attention = None
    
    if decoder_type == "Transformer":
        decoder =  TransformerDecoder(
            vocabs = vocabs,
            d_model =  args['gnn_param']['output_size']+args["global_class_param"]["output_size"],
            nhead = args['transformer_param']['n_head'], 
            num_layers = args['transformer_param']['num_layers'], 
            dim_feedforward=args['transformer_param']['dim_feedforward'], 
            dropout= args['transformer_param']['dropout'],
            device = device,
        ).to(device)
    elif decoder_type == "LSTM":
        decoder = LSTMDecoder(
            vocabs = vocabs, 
            # embed_size = args['gnn_param']['output_size']+args["global_class_param"]["output_size"], 
            embed_size = args["global_class_param"]["output_size"], 
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
        graph_output_size = 0,
        global_output_size = args["global_class_param"]["output_size"],
        hidden_size = args["classifier_param"]["hidden_size"],
        num_class = args["classifier_param"]["num_class"],
        dropout_rate = args["classifier_param"]["dropout_rate"]).to(device)

    return encoder, attention, decoder, global_feature_extractor, classifier 
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
    with open(args["vocab_path"], 'rb') as file:
        vocabs = pickle.load(file)
    vocab_size = len(vocabs)
    #   set path to save checkpoints

    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #   make the dl here
    #   !!!!!!!!!!! Change it back to train
    train_eval_dl,_ = make_dataloader(
        batch_size = args["batch_size"],
        split = "train",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True,
        mode = "eval"
    )
    train_dl,_ = make_dataloader(
        batch_size = args["batch_size"],
        split = "train",
        base_data_path = args["dataset_path"],
        graph_path = args["graph_path"],
        vocab_path = args["vocab_path"],
        shuffle=True,
        num_workers=0,
        load_in_ram = True
    )
    print(f"train loader size {len(train_dl)}")

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


    encoder , attention, decoder,  global_feature_extractor, classifier = model_def(args,DEVICE,vocabs,decoder_type=args["decoder_type"])

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    all_params = list(encoder.parameters()) + list(global_feature_extractor.parameters()) + list(decoder.parameters())
    print(f"----------TOTAL NUMBER OF PARAMETERS: {len(all_params)}-------------")
    if args["optimizer_type"] == "Adam":
        caption_optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"], weight_decay=args["weight_decay"])
        #classifier_optimizer = torch.optim.Adam(params = encoder_classifier_param, lr = args["learning_rate"], weight_decay=args["weight_decay"])
        #optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"])
    elif args["optimizer_type"] == "SGD":
        caption_optimizer = torch.optim.SGD(params=all_params, lr=args["learning_rate"],weight_decay=args["weight_decay"])
        #classifier_optimizer = torch.optim.SGD(params = encoder_classifier_param, lr = args["learning_rate"], weight_decay=args["weight_decay"])
    if "new" in args["vocab_path"]:
        vocab_use = "new"
    else:
        vocab_use = "old"

    torch.autograd.set_detect_anomaly(True)
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
                "num_param":len(all_params),
                "vocab_size":vocab_size,
                "loss":args["loss"],
                "encoder":args["encoder"],
                "decoder":args["decoder_type"],
                "vocab_use":vocab_use
            }
        )
        total_samples = len(train_dl)
        batch_size = args["batch_size"]
        total_step = math.ceil(total_samples / batch_size)
        best_eval_loss = 10
        best_encoder = None
        best_decoder = None
        best_feature_extractor = None
        best_epoch = 0
        print(f"Number of steps per epoch: {total_step} length is {total_samples}")
        current_loss = 0
        current_encoder = None
        current_feature_extractor = None
        current_decoder = None
        for epoch in range(args["epochs"]):

            total_loss = []
            # for batched_idx, batch_data in enumerate(tqdm(train_dl)):
            for step in range(total_step):
                #if args["graph_model_type"] == "Hierarchical":
                cg, tg, assign_mat, caption_tokens, labels, caption, images = next(iter(train_dl))
                #print(f"caption tokens type{type(caption_tokens)} shape {caption_tokens.shape}")
                cg = cg.to(DEVICE)
                tg = tg.to(DEVICE)
                labels = labels.to(DEVICE)
                images = images.to(DEVICE)
                # assign_mat = assign_mat.to(DEVICE)
                caption_tokens = caption_tokens.to(DEVICE) # (batch_size, num_sentences, num_words_in_sentence) num_sentence = 6, num_words = 16

                out = encoder(cg,tg,assign_mat,images) # (batch_size, 1, embedding)
                global_feat = global_feature_extractor(images)


                # merged_feat = torch.cat((out, global_feat), dim=1)
                merged_feat = global_feat
                '''
                merged_feat = global_feat.unsqueeze(1)
                '''
                lstm_out = decoder(merged_feat,caption_tokens)
                pred_matrix = classifier(merged_feat)
                pred_matrix = pred_matrix.to(DEVICE)

                '''
                all_loss = []
                for cap_idx in range(caption_tokens.size(1)):
                    print(f"cap_tok shape at {cap_idx} is {caption_tokens[:,cap_idx,:].shape}")
                    print(f"first ist {lstm_out.view(-1, vocab_size).shape}")
                    all_loss.append(criterion(lstm_out.view(-1, vocab_size),caption_tokens[:,cap_idx,:].reshape(-1)))
                loss = sum(all_loss) / len(all_loss)
                '''
                caption_loss = criterion(lstm_out.view(-1, vocab_size) , caption_tokens.view(-1) )
                #print(f"pred matrix {pred_matrix.shape} labels {labels.shape}")
                #label_loss = criterion(pred_matrix,labels)
                # loss = caption_loss + 0.5*label_loss
                #loss = label_loss
                caption_optimizer.zero_grad()
                #loss = criterion(lstm_out.view(-1, vocab_size) , caption_tokens)
                caption_loss.backward(retain_graph=True)
                #label_loss.backward()
                nn.utils.clip_grad_norm_(all_params, 2.0)
                caption_optimizer.step()

                #classifier_optimizer.step()
                total_loss.append(caption_loss.item())

            mean_loss = np.mean(total_loss)

            del total_loss
            print(f"Training mean loss as {mean_loss} in epoch {epoch}")

            scores,eval_loss,accuracy,f1_score = eval(eval_dl,
                                                      encoder,
                                                      attention,
                                                      decoder,
                                                      global_feature_extractor,
                                                      classifier,
                                                      DEVICE, criterion,vocab_size,labels)
            
            current_loss = eval_loss.item()
            current_encoder = encoder
            current_feature_extractor = global_feature_extractor
            current_decoder = decoder
            if eval_loss.item() < best_eval_loss:
                best_eval_loss = eval_loss.item()
                best_epoch = epoch
                best_encoder = encoder
                best_decoder = decoder
                best_feature_extractor = global_feature_extractor

            eval_output = {
                'train_cap_loss':mean_loss,
                'eval_cap_loss':eval_loss,
                'bleu1':scores['Bleu1'],
                'bleu4':scores['Bleu4'],
                'meteor':scores['METEOR'],
                'rouge':scores['ROUGE_L'],
                'cider':scores['CIDEr'],
                #'spice':scores['SPICE'],
                'accuracy':accuracy,
                'f1_score':f1_score,
            }
            wandb.log(eval_output)

            

            torch.cuda.empty_cache()
        print(f"!!!At epoch [{str(epoch+1)}/{args['epochs']}] evaluate results is {eval_output}")
        
        if args["save_model"]:
            print(f"eval list item is {best_eval_loss}")
            save_model_to_path(args,best_encoder,best_decoder,best_feature_extractor,best_eval_loss,best_epoch)
            save_model_to_path(args,current_encoder,current_decoder,current_feature_extractor,current_loss,args["epochs"]-1)
        # train_scores, train_loss, train_accuracy, train_f1_score = eval(train_eval_dl,
        #                                             encoder,
        #                                             attention,
        #                                             decoder,
        #                                             global_feature_extractor,
        #                                             classifier,
        #                                             DEVICE,criterion,vocab_size,labels)
        # print(f"train loss is {train_loss} for epoch {epoch}")

        if args["save_model"]:
            print(f"eval list item is {best_eval_loss}")
            save_model_to_path(args,best_encoder,best_decoder,best_feature_extractor,best_eval_loss,best_epoch)
        
        scores,_,accuracy,f1_score = eval(test_dl,encoder,attention,decoder, 
                                            global_feature_extractor,classifier,DEVICE,criterion,vocab_size,labels,eval = False)
        # train_eval_output = {
        #     'train_bleu1':train_scores['Bleu1'],
        #     'train_bleu4':train_scores['Bleu4'],
        #     'train_meteor':train_scores['METEOR'],
        #     'train_rouge':train_scores['ROUGE_L'],
        #     'train_cider':train_scores['CIDEr'],
        #     #'spice':scores['SPICE'],
        #     'train_accuracy':train_accuracy,
        #     'train_f1_score':train_f1_score,
        # }
        # print("Training Set as eval mode")
        # print(train_eval_output)
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
                'accuracy':accuracy,
                'f1_score':f1_score,
            }
        print("Testing data output: ")
        print(test_output)


    else:
        #   Only run the Test set
        print("test")




if __name__ == "__main__":
    main()



'''

- Train the captioning, feature extraction and caption generation first. Saved the Encoder, Global Feat Extarction and Decoder
- Then Train the Classifier 

'''