import torch
import torch.nn as nn
import math


import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocabs, d_model, nhead, num_layers,dim_feedforward=2048, dropout=0.1,device = "cpu"):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.vocabs = vocabs
        self.vocab_size = len(vocabs)
        self.device = device
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.word_embedding = nn.Embedding( self.vocab_size , self.d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers
        )
        self.fc_out = nn.Linear(d_model, self.vocab_size)
        self.start_token = self.vocabs.word2idx['<start>'] # Default
        self.end_token = self.vocabs.word2idx['<end>']



    '''
    memory: input embedding : (batch_size, embedding_size)
    tgt: tgt_size: (batch_size, max_len)
    '''
    def forward(self, memory, tgt):
        seq_len = tgt.shape[1]

        tgt_mask = nn.Transformer().generate_square_subsequent_mask(seq_len).to(self.device)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        memory = memory.unsqueeze(0)
        tgt = tgt.permute(1,0,2)

        output = self.transformer_decoder(tgt, memory, tgt_mask = tgt_mask)


        output = self.fc_out(output)

        return output
    


    '''
    memory: input embedding : (batch_size, embedding_size)
    '''
    def predict(self,memory, max_len = 90):
        batch_size = memory.shape[0]
        outputs_tensor = torch.zeros((batch_size, max_len, self.vocab_size)).to(self.device)

        generated_words = torch.LongTensor([self.start_token]*batch_size).unsqueeze(1).to(self.device)
        outputs_tensor[:,0,self.start_token] = 1
        for i in range(max_len-1):
     

            with torch.no_grad():
                output = self.forward(memory, generated_words)
                print(output.shape)
            outputs_tensor[:,i+1,:] = output[-1,:,:]

            next_word = torch.argmax(output[-1,:,:],dim = 1).unsqueeze(1)


            generated_words = torch.cat((generated_words, next_word), dim=1)
            print(f"gen words {generated_words}")

            if (next_word == self.end_token).all():
                break

        return generated_words,outputs_tensor


if __name__ == "__main__":
    from Vocabulary import Vocabulary
    import pickle
    with open("new_vocab_bladderreport.pkl", 'rb') as file:
        vocabs = pickle.load(file)
    decoder = TransformerDecoder( vocabs = vocabs, d_model = 512,        
        nhead = 2, 
        num_layers = 3, 
        dim_feedforward=2048, 
        dropout=0.2)
    x = torch.rand(2,512) 
    cap_tok = torch.rand(2,90) 
    # decoder(x,cap_tok.long())
    out,out_ten = decoder.predict(x,5)
    # print(out.shape)

    print(out)
    print(out_ten.shape)
    # print(out[0])
    # print(out[1])
