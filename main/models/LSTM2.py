import torch.nn as nn
import torch


class LSTMDecoder(nn.Module):
    def __init__(self,vocabs,embed_size, batch_size, hidden_size,bi_direction, device = "cpu",dropout = 0.5, num_layers = 1):
        super().__init__()
        assert dropout <= 1
        self.dropout_rate = dropout
        self.vocabs = vocabs
        self.vocab_size = len(self.vocabs)
        self.device = device
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.bi_direction = bi_direction
        self.direction_weight = 2 if self.bi_direction else 1
        self.dropout = nn.Dropout(p = self.dropout_rate)
        self.word_embedding = nn.Embedding( self.vocab_size , self.embed_size)
        self.start_token = self.vocabs.word2idx['<start>'] # Default
        self.end_token = self.vocabs.word2idx['<end>']
        print(f"pad is {self.vocabs.word2idx['<pad>']}")

        # self.lstm = nn.LSTM(self.embed_size, 
        #                     self.hidden_size, 
        #                     self.num_layers, 
        #                     batch_first = True, 
        #                     bidirectional = self.bi_direction) # tensor shape (batch, seq, feature)  when batch_first = True
        print(f"self.bi {self.bi_direction} hidd {self.hidden_size}")
        self.lstm = nn.LSTM(self.embed_size, 
                            self.hidden_size, 
                            self.num_layers,
                            bias = True, 
                            batch_first=True, 
                            bidirectional=self.bi_direction,
                            dropout = self.dropout_rate)
        self.linear = nn.Linear(self.hidden_size*self.direction_weight, self.vocab_size)
        self.init_h = nn.Linear(self.embed_size,self.hidden_size)
        self.init_c = nn.Linear(self.embed_size,self.hidden_size)

        torch.nn.init.kaiming_normal_(self.linear.weight) #linear init
        torch.nn.init.kaiming_normal_(self.word_embedding.weight) #embedding

    def init_hidden(self,features,batch_size = None):

        if batch_size is None:
            batch_size = self.batch_size
        
        h0 = self.init_h(features.permute(1,0,2))
        #return (h0.to(self.device),torch.zeros( self.num_layers * self.direction_weight , batch_size , self.hidden_size  ).to(self.device))
        return ( torch.zeros(self.num_layers * self.direction_weight , batch_size , self.hidden_size  ).to(self.device),
        torch.zeros( self.num_layers * self.direction_weight , batch_size , self.hidden_size  ).to(self.device) )

    '''
    
    Do it iteratively.
    '''
    def forward(self,features,captions,mask = None):
        print(f"input features {features.shape}")
        features = features.unsqueeze(1)
        if self.batch_size != features.shape[0]:
            self.batch_size = features.shape[0] 
        # batch_size = captions.shape[0]
        # seq_length = captions.shape[1]
        # print(f"caption forward {captions.shape}")
        captions = captions[:,:-1]
        embeds = self.word_embedding( captions)
        h_0, c_0 = self.init_hidden(features)
        # print(f"h0 is {h_0}")

        # print(f"before cat, feature {features.shape} and embeds is {embeds.shape}")
        inputs = torch.cat(( features, embeds ), dim =1)
        # print(f"after cat, feature {features.shape} and embeds is {embeds.shape}")
        #embeddings,_ = self.lstm(inputs,(h_0,c_0))
        embeddings,_ = self.lstm(inputs,(h_0,c_0)) # GRU only
        # print(f"after lstm embedding {embeddings.shape}")
        # caption shape (bs,90) -> embeds shape (bs,89,1028)
        # feature shape (bs,1,1028) 
        # Concat 1 & 2 gives (bs, 90,1028)
        # tput shape is (bs,90,vocab_size)

        outputs = self.dropout(self.linear(embeddings))
        #outputs = self.linear(self.dropout(embeddings))
        
        return outputs

    # def postprocess_output(self,final_output, end_token=2, pad_token=1):
    #     for sentence in final_output:
    #         end_token_idx = (sentence == end_token).nonzero(as_tuple=True)[0]
    #         if len(end_token_idx) > 0:
    #             first_end_token_idx = end_token_idx[0].item()
    #             sentence[first_end_token_idx + 1:] = pad_token
    #     return final_output

    def predict(self, inputs, max_len=50):   
        inputs = inputs.unsqueeze(1)     
        batch_size = inputs.shape[0]  
        final_output = torch.zeros(batch_size, max_len, dtype=torch.long).to(self.device)       
        hidden = self.init_hidden(inputs,batch_size)
        outputs_tensor = torch.zeros((batch_size, max_len, self.vocab_size)).to(self.device)
        for idx in range(max_len-1):
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.dropout(self.linear(lstm_out))
 
            outputs = outputs.squeeze(1) 
            outputs_tensor[:,idx,:] = outputs
            
 
            _, max_idx = torch.max(outputs, dim=1) 

            final_output[:,idx] = max_idx
            if (idx >= max_len ):
                break
            
            inputs = self.word_embedding(max_idx) 
            inputs = inputs.unsqueeze(1)    
        #print(final_output)
        #print("##########")
        # final_output = self.postprocess_output(final_output) 
        #print(post_output)
        return final_output,outputs_tensor 



if __name__ == "__main__":
    from Vocabulary import Vocabulary
    import pickle
    with open("new_vocab_bladderreport.pkl", 'rb') as file:
        vocabs = pickle.load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = LSTMDecoder(
            vocabs = vocabs, 
            embed_size = 512, 
            hidden_size = 128,  
            batch_size= 8, 
            bi_direction = False,
            device = device,
            dropout = 0.5,
            num_layers = 1
        ).to(device)
    x = torch.rand(2,512).to(device)
    cap = torch.rand(2,80).to(device)
    print(decoder)
    # Define the integer range you want (e.g., [0, 10))
    int_range_low = 0
    int_range_high = 10

    # Scale and convert to integers
    cap = (cap * (int_range_high - int_range_low) + int_range_low).to(torch.int32)
    print(f"feature shape {x.shape} cap shape {cap.shape}")
    out= decoder(x,cap)
    print(f"out shape {out.shape}")
    print(f"----predict-----")
    out,out_ten = decoder.predict(x)
    
    # print(f"out shape {out}")

    # print(f"out tensor shape {out_ten}")