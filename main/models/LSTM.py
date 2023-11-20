import torch.nn as nn
import torch


class LSTMDecoder(nn.Module):
    def __init__(self,vocabs,embed_size, batch_size, hidden_size,bi_direction, device = "cpu",dropout = 0.5, num_layers = 1):
        super().__init__()
        assert dropout <= 1
        self.dropout = dropout
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
        self.dropout = nn.Dropout(p = self.dropout)
        self.word_embedding = nn.Embedding( self.vocab_size , self.embed_size)
        self.start_token = self.vocabs.word2idx['<start>'] # Default
        self.end_token = self.vocabs.word2idx['<end>']

        self.lstm = nn.LSTM(self.embed_size, 
                            self.hidden_size, 
                            self.num_layers, 
                            batch_first = True, 
                            bidirectional = self.bi_direction) # tensor shape (batch, seq, feature)  when batch_first = True
        
        #self.lstm = nn.GRU(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*self.direction_weight, self.vocab_size)


    def init_hidden(self,batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
       
        return ( torch.zeros(self.num_layers * self.direction_weight , batch_size , self.hidden_size  ).to(self.device),
        torch.zeros( self.num_layers * self.direction_weight , batch_size , self.hidden_size  ).to(self.device) )

    '''
    
    Do it iteratively.
    '''
    def forward(self,features,captions):
        features = features.unsqueeze(1)
        if self.batch_size != features.shape[0]:
            self.batch_size = features.shape[0] 
        batch_size = captions.shape[0]
        seq_length = captions.shape[1]
        
        captions = captions[:,:-1]

        h_0, c_0 = self.init_hidden()
 
        embeds = self.word_embedding( captions)
        #print(f"before cat, feature {features.shape} and embeds is {embeds.shape}")
        inputs = torch.cat( ( features, embeds ) , dim =1)
         
        embeddings,_ = self.lstm(inputs,(h_0,c_0))
        #print(f"after lstm embedding {embeddings.shape}")
        # caption shape (bs,90) -> embeds shape (bs,89,1028)
        # feature shape (bs,1,1028) 
        # Concat 1 & 2 gives (bs, 90,1028)
        # tput shape is (bs,90,vocab_size)
        outputs = self.dropout(self.linear(embeddings))
        
        return outputs
        '''
        # 5 caption all together, learn together at once

        for cap_idx in range(captions.size(1)):
            h_0, c_0 = self.init_hidden()
            single_caption = captions[:,cap_idx,:]
            single_caption = single_caption[:,:-1]


            embeds = self.word_embedding(single_caption)
            # print(f"before cat, feature {features.shape} and embeds is {embeds.shape}")
            inputs = torch.cat( ( features, embeds ) , dim =1  ) 
            
            embeddings,_ = self.lstm(inputs,(h_0,c_0))

            outputs = self.dropout(self.linear(embeddings))
        print(f"output in lstm shape is {outputs.shape}")
        return outputs
        '''

    def predict(self, inputs, max_len=50):   
        inputs = inputs.unsqueeze(1)     
        batch_size = inputs.shape[0]  
        final_output = torch.zeros(batch_size, max_len, dtype=torch.long)       
        hidden = self.init_hidden(batch_size) 
        # print(f"self start tok {self.start_token}")
        final_output[:,0] = self.start_token
        # print(final_output[0])
        outputs_tensor = torch.zeros((batch_size, max_len, self.vocab_size)).to(self.device)
        outputs_tensor[:,0,self.start_token] = 1
        for idx in range(max_len-1):
            lstm_out, hidden = self.lstm(inputs, hidden) 
            #outputs = self.linear(self.dropout(lstm_out))
            outputs = self.linear(lstm_out)
            # print(f"outputs after linear is {outputs.shape}")
            outputs = outputs.squeeze(1) 
            outputs_tensor[:,idx+1,:] = outputs
            
 
            _, max_idx = torch.max(outputs, dim=1) 
            #print(f"idx = {idx} out ten {outputs_tensor.shape} and {outputs.shape}")
            outputs_tensor[:, idx+1, :] = outputs
            # final_output.append(max_idx.cpu().numpy()[0].item())   
            #print(f"Final output have shape {final_output.shape} with max_isx {max_idx.shape}") 
            # torch.cat((final_output,max_idx.cpu()),dim = 1)         
            final_output[:,idx] = max_idx.cpu()
            if (idx >= max_len ):
                break
            
            inputs = self.word_embedding(max_idx) 
            inputs = inputs.unsqueeze(1)     
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
            hidden_size = 256,  
            batch_size= 8, 
            bi_direction = False,
            device = device,
            dropout = 0.5,
            num_layers = 1
        )
    x = torch.rand(8,512) 
    out,out_ten = decoder.predict(x)
    print(out.shape)
    print(out_ten.shape)
    
