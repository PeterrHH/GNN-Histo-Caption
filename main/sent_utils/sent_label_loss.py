import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def get_single_sent_label_loss(pred,gt,key_words,key_word_weight):
    '''
    pred ([1,5])
    gt ([5])
    '''
    #loss = []
    loss = 0
    for idx,value in enumerate(pred):
 
        pred = F.one_hot(value,len(key_words[idx])+1).float().view(1,-1)
        target = torch.tensor([torch.tensor(gt[idx])])
        weights = torch.FloatTensor(list(key_word_weight[idx]))
        #loss.append(nn.CrossEntropyLoss(weight = weights,reduction = "mean")(pred,target))
        loss += nn.CrossEntropyLoss(weight = weights,reduction = "mean")(pred,target)
    # return np.mean(loss)
    return loss


def get_eval_sent_label_loss(pred,gt,key_word,key_word_weight):
    loss_list = []
    for idx in range(5):
        loss_list.append(get_single_sent_label_loss(pred,gt[idx],key_word,key_word_weight))
    return np.mean(loss_list)

def get_batch_sent_label_loss(pred,gt,key_words,key_word_weight,phase):
    # train, and eval/test have different strategies
    bs = pred.shape[0]
    loss_list = []
    print()
    print(f"pred len {pred.shape} and len gt {gt.shape} bs is {bs}")
    for idx in range(bs):
        #print(idx)
        if phase == "train":
            loss_list.append(get_single_sent_label_loss(pred[idx],gt[idx],key_words,key_word_weight))
        else:
            loss_list.append(get_eval_sent_label_loss(pred[idx],gt[idx],key_words,key_word_weight))
        
    return np.mean(loss_list)

