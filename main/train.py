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
from torch.optim.lr_scheduler import StepLR

import yaml
import sys
sys.path.append('../histocartography/histocartography')
sys.path.append('../histocartography')

from evaluation import Scorer
from dataloader import make_dataloader,dataset_to_loader

import wandb

from argparse import ArgumentParser
from models.Attention import EncoderLayer
from Vocabulary import Vocabulary
from models.Graph_Model import GNNEncoder
#from models.LSTM import LSTMDecoder, Beam_LSTMDecoder
from models.LSTM2 import LSTMDecoder
from models.GlobalFeatureExtractor import GlobalFeatureExtractor
from models.Classifier import Classifier
from models.Transformer import TransCapDecoder
from data_plotting import violin_plot
from torch.utils.data import WeightedRandomSampler

import sent_utils


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

MAX_LEN = 80

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


def calc_acc_f1(gt_label,pred_matrix,batch_size,device):
    pred_label = torch.argmax(pred_matrix, dim=1)
    #print(f"shape of pred_matrix is {pred_matrix.shape} pred label {pred_label.shape} gt_label {gt_label.shape} ")
    #print(f"------First label {pred_label} and gt {gt_label[0]}------")

    # correct_predictions = (pred_label == gt_label).sum().item()
    # accuracy = correct_predictions / batch_size
    accuracy = accuracy_score(pred_label.cpu().numpy(), gt_label.cpu().numpy())
    f1 = f1_score(gt_label.cpu().numpy(), pred_label.cpu().numpy(), average='weighted') 
    return accuracy, f1

def get_clip_inference(model,cg,tg,assign_mat,images,trainable = True):
    graph_out = model.graph_encoder(cg,tg,assign_mat,images)
    global_feat = model.feature_extractor(images)
    merged_feat = torch.cat((graph_out, global_feat), dim=1)
    image_embeddings = model.image_projection(merged_feat)

    return image_embeddings

def eval(eval_loader,if_encoding,image_encoding,
         encoder,attention,decoder, global_feature_extractor, classifier,
         device,
         caption_criterion,class_criterion, vocab_size, labels,eval = True) :
    batch_size = eval_loader.batch_size
    total_samples = len(eval_loader.dataset)
    total_step = math.ceil(total_samples / batch_size)

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
    caption_loss = []
    sent_lbl_loss = []
    label_loss = []

    for step in tqdm(range(total_step)):
        pred_dict = {}
        cap_dict = {}
        cg, tg, assign_mat, caption_tokens, labels, captions, images,attention_masks,gt_lab = next(iter(eval_loader))
        cg = cg.to( device)
        tg = tg.to(device)
        images =  images.to(device)
        labels = labels.to(device)
        # encoder , decoder, attention = encoder.to(device) , decoder.to(device), attention.to(device)
        caption_tokens = caption_tokens.to(device)
        attention_masks = attention_masks.to(device)
        with torch.no_grad():
            if if_encoding:
                merged_feat = get_clip_inference(image_encoding,cg,tg,assign_mat,images)
                # merged_feat = image_encoding(cg,tg,assign_mat,images,caption_tokens,attention_masks)
            else:
                out = encoder(cg,tg,assign_mat,images)
                global_feat = global_feature_extractor(images)
                merged_feat = torch.cat((out, global_feat), dim=1)
                #merged_feat = global_feat
            lstm_out, lstm_out_tensor = decoder.predict(merged_feat,MAX_LEN)
            #lstm_out_tensor = decoder.predict(merged_feat,caption_tokens,attention_masks)
            # print(f"lstm out ten shape {lstm_out_tensor.shape}")
            # _,lstm_out = torch.max(lstm_out_tensor,dim = 2)
            # print(f"lstm out shape {lstm_out.shape}")
            pred_matrix = classifier(merged_feat)
            pred_matrix = pred_matrix.to(device)
            
        #   Evaluate
        if eval:
            all_eval_loss = []
            all_class_loss = []
            for cap_idx in range(caption_tokens.size(1)):
                #print(f"lsmt out ten {lstm_out_tensor.shape} cap tok cap_idx {caption_tokens[:,cap_idx,:].shape}")
                eval_cap_loss = caption_criterion(lstm_out_tensor.view(-1, vocab_size),caption_tokens[:,cap_idx,:].reshape(-1))

                eval_class_loss = class_criterion(pred_matrix,labels)
                all_eval_loss.append(eval_cap_loss.item())
                all_class_loss.append(eval_class_loss.item())
            # caption_loss = caption_criterion(lstm_out_tensor.contiguous().view(-1, vocab_size),caption_tokens.reshape(-1))
            # label_loss = class_criterion(pred_matrix,labels)
            #caption_loss.append(eval_cap_loss.item())
            # eval_loss = sum(all_eval_loss) / len(all_eval_loss)
            eval_loss = np.mean(all_eval_loss)
            caption_loss.append(eval_loss.item())
            eval_label_loss = np.mean(all_class_loss)
            label_loss.append(eval_label_loss.item())

        else:
            eval_loss = None
            eval_label_loss = None
    
        accuracy,f1_score = calc_acc_f1(labels,pred_matrix,batch_size,device)
        #print(f"lstm out shape in eval{lstm_out.shape}")
        pred_dict,cap_dict = sent_utils.embed2sentence(lstm_out,eval_loader,captions,pred_dict,cap_dict,"eval")

        '''
        pred_lab = sent_utils.get_all_pred_label(pred_dict,sent_utils.key_words_name,sent_utils.key_words)
        #print(f"pred lab in eval {pred_lab.shape}")
        single_sent_lbl_loss = sent_utils.get_batch_sent_label_loss(pred_lab,gt_lab,sent_utils.key_words,sent_utils.key_word_weights,"eval")
        sent_lbl_loss.append(single_sent_lbl_loss.item())
        '''
        # eval_loss += sent_lbl_loss
        #print(f"cap dict len{len(cap_dict)} pred dict {len(pred_dict)}")
        scorer = Scorer(cap_dict,pred_dict)
        # if eval:
        scores = scorer.compute_scores()
        if eval is False:
            for key,value in scores.items():
                test_output[key].append(value[0])

    if eval:
        # compute only mean
        test_output = {key: value[0] for key, value in scores.items()}
        eval_sent_loss = np.mean(caption_loss)
        # eval_sent_loss = caption_loss

        #eval_sent_lbl_loss = np.mean(sent_lbl_loss)
        eval_sent_lbl_loss = 0
        eval_loss = eval_sent_loss+ 0*eval_sent_lbl_loss
        # eval_label_loss = label_loss
        eval_label_loss = np.mean(label_loss)
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

    return test_output,eval_loss,eval_label_loss,accuracy,f1_score,eval_sent_loss,eval_sent_lbl_loss

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
    decoder embed size 
    '''
    decoder = LSTMDecoder(
        vocab_size = vocabs, 
        embed_size = 256, 
        hidden_size = 256,  
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

'''
decoder transformer/LSTM
'''
def model_def(args,device,vocabs,decoder_type = "Transformer",image_embed_path = None):
    '''
    decoder: transformer/lstm
    '''
        #   Define Model, Loss and 


    # attention = EncoderLayer(d_model = args['gnn_param']['output_size']+args['global_class_param']['output_size'], 
    #     nhead = 4, 
    #     dim_feedforward = 1024, 
    #     dropout = 0.2).to(device)
    image_embed_model = None
    encoder = None
    global_feature_extractor = None
    attention = None
    if decoder_type == "Transformer":
        decoder =  TransCapDecoder(
            vocabs = vocabs,
            embed_size =  args['gnn_param']['output_size']+args["global_class_param"]["output_size"],
            nhead = args['transformer_param']['n_head'], 
            num_layers = args['transformer_param']['num_layers'], 
            dim_feedforward= args['transformer_param']['dim_feedforward'], 
            dropout= args['transformer_param']['dropout'],
            device = device,
        ).to(device)
    elif decoder_type == "LSTM":
        decoder = LSTMDecoder(
            vocabs = vocabs, 
            #embed_size = args["global_class_param"]["output_size"], 
            embed_size = args['gnn_param']['output_size']+args["global_class_param"]["output_size"], 
            #embed_size = args['gnn_param']['output_size'], 
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


    classifier = Classifier(
        graph_output_size = args['gnn_param']['output_size'],
        global_output_size = args["global_class_param"]["output_size"],
        hidden_size = args["classifier_param"]["hidden_size"],
        num_class = args["classifier_param"]["num_class"],
        dropout_rate = args["classifier_param"]["dropout_rate"]).to(device)
   
    if image_embed_path is None:
        encoder = GNNEncoder(
            args = args,
            cg_layer = args['gnn_param']['cell_layers'], 
            tg_layer = args['gnn_param']['tissue_layers'],
            aggregate_method = args['gnn_param']['aggregate_method'], 
            input_feat = 514,
            hidden_size = args['gnn_param']['hidden_size'],
            output_size = args['gnn_param']['output_size'],
        ).to(device)
        global_feature_extractor = GlobalFeatureExtractor(
            hidden_size = args["global_class_param"]["hidden_size"],
            output_size = args["global_class_param"]["output_size"],
            dropout_rate = args["global_class_param"]["dropout_rate"]).to(device)
    else:
        print(f"LOADING IMAGE EMBED")
        image_embed_model = torch.load(image_embed_path, map_location=device)
        
        for name, param in image_embed_model.named_parameters():
    #if 'fc' not in name:  # or any other condition based on layer names
            param.requires_grad = args["trainable_embedding"]
        trainable = any(param.requires_grad for param in image_embed_model.parameters())
        print("Model is trainable:", trainable)
    return image_embed_model, encoder , attention, decoder,  global_feature_extractor, classifier
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
    #train_dl = get_sample_samplier(train_dataset,args["batch_size"])
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

    if args["load_embedding"]:
        print(f"USING THE LOAD EMBEDDING")
        image_encoding, encoder , attention, decoder,  global_feature_extractor, classifier = model_def(args,DEVICE,vocabs,decoder_type=args["decoder_type"],image_embed_path = args["embed_model_path"])
        all_params = list(image_encoding.parameters()) + list(decoder.parameters())
    else:
        image_encoding, encoder , attention, decoder,  global_feature_extractor, classifier = model_def(args,DEVICE,vocabs,decoder_type=args["decoder_type"])
        all_params = list(encoder.parameters()) + list(global_feature_extractor.parameters()) + list(decoder.parameters())
        #all_params = list(decoder.parameters())
    all_params = list(encoder.parameters()) + list(global_feature_extractor.parameters()) + list(decoder.parameters())
    #caption_criterion = nn.CrossEntropyLoss(ignore_index = vocabs.word2idx['<pad>']).cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss(ignore_index = vocabs.word2idx['<pad>'])
    caption_criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    #sent_label_criterion = nn.CrossEntropyLoss(weight = sent_utils.key_word_weights,reduction = 'mean').cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    class_criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    
    encoder_classifier_param = list(classifier.parameters())
    print(f"----------TOTAL NUMBER OF PARAMETERS: {len(all_params)}-------------")
    if args["optimizer_type"] == "Adam":
        #caption_optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"])
        if args["decoder_type"] == "Transformer":
            caption_optimizer = torch.optim.Adam([
                {'params': list(encoder.parameters()) + list(global_feature_extractor.parameters()) , 'lr': 0.001,'weight_decay':0.001},
                {'params': list(decoder.parameters()), 'lr': args["learning_rate"],'weight_decay':args["weight_decay"]}
            ])
        else:
            caption_optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"], weight_decay=args["weight_decay"])
            #caption_optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"], weight_decay=args["weight_decay"])
        #caption_optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"],weight_decay=args["weight_decay"])
        #caption_optimizer = torch.optim.Adam(params = all_params, lr= args["learning_rate"])
        scheduler = StepLR(caption_optimizer, step_size=20, gamma=0.9)
        classifier_optimizer = torch.optim.Adam(params = encoder_classifier_param, lr = args["learning_rate"], weight_decay=args["weight_decay"])

    elif args["optimizer_type"] == "SGD":
        caption_optimizer = torch.optim.SGD(params=all_params, lr=args["learning_rate"],weight_decay=args["weight_decay"],momentum = 0.9)
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
        patience = 10
        counter = 0
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
            total_class_loss = []
            # for batched_idx, batch_data in enumerate(tqdm(train_dl)):
            for step in range(total_step):
                #if args["graph_model_type"] == "Hierarchical":
                cg, tg, assign_mat, caption_tokens, labels, caption, images,attention_masks,gt_lab = next(iter(train_dl))
                #print(f"caption tokens type{type(caption_tokens)} shape {caption_tokens.shape}")
                # print(caption_tokens[0])
                cg = cg.to(DEVICE)
                tg = tg.to(DEVICE)
                labels = labels.to(DEVICE)
                images = images.to(DEVICE)
                # assign_mat = assign_mat.to(DEVICE)
                attention_masks = attention_masks.to(DEVICE)
                caption_tokens = caption_tokens.to(DEVICE) # (batch_size, num_sentences, num_words_in_sentence) num_sentence = 6, num_words = 16
                if args["load_embedding"]:
                    merged_feat = get_clip_inference(image_encoding,cg,tg,assign_mat,images)
                    #merged_feat = image_encoding(cg,tg,assign_mat,images,caption_tokens,attention_masks)
                else:
                    out = encoder(cg,tg,assign_mat,images) # (batch_size, 1, embedding)
                    global_feat = global_feature_extractor(images)
            
                    
                    merged_feat = torch.cat((out, global_feat), dim=1)
                    #merged_feat = global_feat

                #print(f"merged_feat shape before decoder {merged_feat.shape} cap tok {caption_tokens.shape}")
                print(f"------Caption token in Train--------")
                print(caption_tokens[0,:])
                lstm_out = decoder(merged_feat,caption_tokens,attention_masks)
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
                pred_dict = {}
                cap_dict = {}
                #pred_dict,cap_dict = sent_utils.embed2sentence(lstm_out,train_dl,caption,pred_dict,cap_dict,"train")
                #pred_lab = sent_utils.get_all_pred_label(pred_dict,sent_utils.key_words_name,sent_utils.key_words)
                # print(f"lsmt out shape {lstm_out.shape} cap tok shape {caption_tokens.shape}")
                # print(f"vocab size is {vocab_size}")
                # print(f"lstm reshape {lstm_out.contiguous().view(-1, vocab_size).shape}")
                # print(f"cap tok reshape {caption_tokens.view(-1).shape}")
                # print(f"lstm_out 50: {lstm_out.shape}")
                # print(f"caption_tok: {caption_tokens.shape}")
                # print(lstm_out.contiguous().view(-1, vocab_size)[])
                # print("!!!!!!!!!!!!!!!!!!!")
                caption_loss = caption_criterion(lstm_out.view(-1, vocab_size) , caption_tokens.view(-1) )
                # print(f"LOSS is {caption_loss}")
                # print(f"----------")
                #sent_lbl_loss = sent_utils.get_batch_sent_label_loss(pred_lab,gt_lab,sent_utils.key_words,sent_utils.key_word_weights,"train")
                sent_lbl_loss = 0
                final_loss = caption_loss 
                # final_loss = sent_lbl_loss
                label_loss = class_criterion(pred_matrix,labels)
                # loss = caption_loss + 0.5*label_loss
                #loss = label_loss
                caption_optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                caption_loss.backward(retain_graph=True)
                
                label_loss.backward()
                nn.utils.clip_grad_norm_(all_params, 1.0)
                caption_optimizer.step()

                classifier_optimizer.step()
                # total_loss.append(caption_loss.item())
                total_loss.append(final_loss.item())
                total_class_loss.append(label_loss.item())
            scheduler.step()
            mean_loss = np.mean(total_loss)
            mean_label_loss = np.mean(total_class_loss)
            del total_loss
            #print(f"Training caption mean loss as {mean_loss} and label loss {mean_label_loss} in epoch {epoch}")
            encoder.eval()
            decoder.eval()
            global_feature_extractor.eval()
            scores,eval_loss,eval_class_loss,accuracy,f1_score,cap_loss,sent_lbl_loss = eval(eval_dl,if_encoding = args["load_embedding"],image_encoding = image_encoding,
                                                      encoder = encoder,
                                                      attention = attention,
                                                      decoder = decoder,
                                                      global_feature_extractor = global_feature_extractor,
                                                      classifier = classifier,
                                                      device = DEVICE, caption_criterion = caption_criterion,
                                                      class_criterion = class_criterion,vocab_size = vocab_size,labels = labels)
            
            current_loss = eval_loss.item()
            current_encoder = encoder
            current_feature_extractor = global_feature_extractor
            current_decoder = decoder


            eval_output = {
                'train_cap_loss':mean_loss,
                'eval_cap_loss':eval_loss,
                'train_class_loss':mean_label_loss,
                'eval_class_loss':eval_class_loss,
                'bleu1':scores['Bleu1'],
                'bleu4':scores['Bleu4'],
                'meteor':scores['METEOR'],
                'rouge':scores['ROUGE_L'],
                'cider':scores['CIDEr'],
                #'spice':scores['SPICE'],
                'accuracy':accuracy,
                'f1_score':f1_score,
                # 'cap_loss':cap_loss,
                # 'sent_lbl_loss':sent_lbl_loss,
            }
            encoder.train()
            decoder.train()
            global_feature_extractor.train()
            wandb.log(eval_output)
            if args["save_model"] is True:
                if args["decoder_type"] == "Transformer" :
                    if epoch % 5 == 0:
                        save_model_to_path(args,encoder,decoder,global_feature_extractor,eval_loss.item(),epoch)
                        torch.cuda.empty_cache()
                else:
                    if epoch >= 50 and epoch % 5 == 0:
                            save_model_to_path(args,encoder,decoder,global_feature_extractor,eval_loss.item(),epoch)
                            torch.cuda.empty_cache()
            if eval_loss.item() < best_eval_loss:
                best_eval_loss = eval_loss.item()
                best_epoch = epoch
                best_encoder = encoder
                best_decoder = decoder
                counter = 0
                best_feature_extractor = global_feature_extractor
            else:
                counter += 1
                #   Do early stopping a bit later
                if counter >= 10 and epoch > 140:
                        save_model_to_path(args,best_encoder,best_decoder,best_feature_extractor,best_eval_loss,best_epoch)
                        torch.cuda.empty_cache()
                        break
                # save_model_to_path(args,current_encoder,current_decoder,current_feature_extractor,current_loss,args["epochs"]-1)


            

            torch.cuda.empty_cache()
        print(f"!!!At epoch [{str(epoch+1)}/{args['epochs']}] evaluate results is {eval_output}")
        
        # if args["save_model"]:
        #     print(f"eval list item is {best_eval_loss}")
        #     save_model_to_path(args,best_encoder,best_decoder,best_feature_extractor,best_eval_loss,best_epoch)
        #     save_model_to_path(args,current_encoder,current_decoder,current_feature_extractor,current_loss,args["epochs"]-1)
        # # train_scores, train_loss, train_accuracy, train_f1_score = eval(train_eval_dl,
        # #                                             encoder,
        # #                                             attention,
        # #                                             decoder,
        # #                                             global_feature_extractor,
        # #                                             classifier,
        # #                                             DEVICE,criterion,vocab_size,labels)
        # # print(f"train loss is {train_loss} for epoch {epoch}")

        # if args["save_model"]:
        #     print(f"eval list item is {best_eval_loss}")
        #     save_model_to_path(args,best_encoder,best_decoder,best_feature_extractor,best_eval_loss,best_epoch)
        '''

        scores,eval_loss,eval_class_loss,accuracy,f1_score = eval(eval_dl,if_encoding = args["load_embedding"],image_encoding = image_encoding,
                                                    encoder = encoder,
                                                    attention = attention,
                                                    decoder = decoder,
                                                    global_feature_extractor = global_feature_extractor,
                                                    classifier = classifier,
                                                    device = DEVICE, caption_criterion = caption_criterion,
                                                    class_criterion = class_criterion,vocab_size = vocab_size,labels = labels)
        '''
        
        # scores,_,_,accuracy,f1_score = eval(test_dl,best_encoder,attention,best_decoder, 
        #                                     best_feature_extractor,classifier,DEVICE,caption_criterion,class_criterion,vocab_size,labels,eval = False)
        scores,eval_loss,eval_class_loss,accuracy,f1_score,_,_ = eval(
                                                    eval_loader = test_dl,
                                                    if_encoding = args["load_embedding"],
                                                    image_encoding = image_encoding,
                                                    encoder = best_encoder,
                                                    attention = None,
                                                    decoder = best_decoder,
                                                    global_feature_extractor = best_feature_extractor,
                                                    classifier = classifier,
                                                    device = DEVICE, caption_criterion = caption_criterion,
                                                    class_criterion = class_criterion,vocab_size = vocab_size,labels = labels,eval = False)
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
        print(f"Testing data output for the best epoch at {best_epoch}.")
        print(test_output)


    else:
        #   Only run the Test set
        print("test")




if __name__ == "__main__":
    main()
