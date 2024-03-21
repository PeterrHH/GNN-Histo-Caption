import torch

key_words = [
            ['severe','moderate','normal','mild'],
            [['no signs','no nuclear crowding'],['normal','normally'],'mild','severe',['moderate','moderately']],
            ['normal',['not lost','negligibly lost','no loss'],['is completely lost','complete lack of','lack of cellular polarity'],['some degree','not completely lost','partially lost']],
            ['rare',['are frequently','is frequent','are frequent'],'infrequent'],
            ['inconspicuous',['are prominent','prominent nucleoi','is prominent'],'rare'],
        ]


def find_key(key_words,sent):
    for key_idx,key_word in enumerate(key_words):
        if isinstance(key_word,str):
            if key_word in sent:
                return key_idx
        else:
            for i in key_word:
                if i in sent:
                    return key_idx

    # if not found_key_word:
    return key_idx
def get_all_pred_label(sentence_dict,key_words_name,key_words):
    return_list = []
    print(f"type is {type(sentence_dict)} and length {len(sentence_dict)}")
    for sent in list(sentence_dict.values()):
        return_list.append(get_single_pred_sentence_label(sent[0],key_words_name,key_words))
        # print(return_list)
        # print("###########")
    return torch.stack(return_list)


def get_single_pred_sentence_label(sentence,key_words_name,key_words):
    sentence_split = sentence.split('.')

    predict_sent_label = [-1,-1,-1,-1,-1]

    for sent in sentence_split:
        found = False
        for key_idx,key_name in enumerate(key_words_name):

            if isinstance(key_name,str):
                if key_name in sent:

                    predict_sent_label[key_idx] = find_key(key_words[key_idx],sent)
                    found = True
            else:
                for i in key_name:
                    if i in sent:
                        predict_sent_label[key_idx] = find_key(key_words[key_idx],sent)
                        found = True
    find_sent_label = torch.tensor([len(key_words[i]) if v == -1 else v for i,v in enumerate(predict_sent_label) ])
    #print(f"Label find {find_sent_label}")

    return find_sent_label


'''
decode output: lstm out
loader: dataloader
captions: caption loaded from the dataloader
pred_dict: dictionary of prediction  that will be returned
cap_dict: dictionary of caption returned will be returned
phase: "train"/"eval"/"test"
'''
    
def embed2sentence(decode_output, loader, captions, pred_dict, cap_dict,phase):
    # phase = "train"
    for i,caption in enumerate(captions):
        # print(caption)
        # print("----")
        #captions[i]= ' '.join(caption.split()).replace("<pad>", "").replace("<end>", ".").replace("<start>","").replace('<unk>',"").replace("<full-stop>",".")

        if phase == "train":
            st = ' '.join(sent_cap.split()).replace("<pad>", "").replace("<end>", "").replace("<start>","").replace('<unk>',"").replace("<full-stop>",".")
            st = [s.strip().capitalize() for s in st.split('.')]
            st= '. '.join(st).rstrip('.')
            captions[i]= st

        else:
            for i,caption in enumerate(captions):
                for sent_i,sent_cap in enumerate(caption):
                    st = ' '.join(sent_cap.split()).replace("<pad>", "").replace("<end>", "").replace("<start>","").replace('<unk>',"").replace("<full-stop>",".")
                    st = [s.strip().capitalize() for s in st.split('.')]
                    st= '. '.join(st).rstrip('.')
                    captions[i][sent_i] = st

    # if phase == "train":
    #     decode_output,_ = torch.max(decode_output, dim=0)
    j = 0

    for idx,embed in enumerate(decode_output):
        
        sentence = " ".join([loader.dataset.vocab.idx2word[int(idx)] for idx in embed])
        sentence = sentence.replace("<pad>","").replace("<start>","")
        sentence = ' '.join(sentence.split()).replace("<end>", "").strip()
        sentence = ' '.join(sentence.split()).replace(" <full-stop>", ".")

        sentences = [s.strip().capitalize() for s in sentence.split('.')]
        sentence = '. '.join(sentences).rstrip('.')

        if len(pred_dict.keys()) == 0:
            #   Empty
            pred_dict["1"] = [sentence]
            #print(f"at 0, sentence is {captions[idx]}")
            # for i in range(5):
            #     cap_dict[str(int(i)+1)] = captions[idx]
            print(f"------embed---------")
            print(embed)
            print("---------------pred---------")
            print(sentence)
            print("---------------gt-----------")
            print(captions[idx][0])
            cap_dict["1"] = captions[idx]
        
            pass
        else:
            pred_dict[str(len(pred_dict)+1)] = [sentence]
            cap_dict[str(len(cap_dict)+1)] = captions[idx]
    
    return pred_dict,cap_dict