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
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.word_embedding = nn.Embedding( self.vocab_size , self.d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.start_token = 119 # Default

    def forward(self, memory, tgt):
        # print(f"memory shape {memory.shape} tgt shape {tgt.shape}")
        #print(f"    min index in tgt: {torch.min(tgt)}, max index in tgt: {torch.max(tgt)}")
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        #print(f"    emb, tgt shape {tgt.shape}")
        tgt = self.pos_encoder(tgt)
        #print(f"    pos_encoder, tgt shape {tgt.shape} mem shape {memory.shape}")
        #print(f"    min index in tgt: {torch.min(tgt)}, max index in tgt: {torch.max(tgt)}")
        memory = memory.unsqueeze(0)
        tgt = tgt.permute(1,0,2)
        print(f"    pos_encoder, tgt shape {tgt.shape} mem shape {memory.shape}")
        output = self.transformer_decoder(tgt, memory)

        #print(f"    transformer, tgt shape {output.shape}")
        output = self.fc_out(output)
        #print(f"    fc out, tgt shape {output.shape}")
        return output
    

    def predict(self,memory, max_len = 90):
        batch_size = memory.shape[0]
        current_token = [self.start_token]*batch_size
        memory = memory.unsqueeze(0)

        final_output = torch.zeros(batch_size, max_len, dtype=torch.long)  
        print(f"fianl shape {final_output.shape}")     
        final_output[:,0] = torch.LongTensor(current_token)
 
        outputs_tensor = torch.zeros((batch_size, max_len, self.vocab_size))
        outputs_tensor[:, 0, 119] = 1
        #outputs_tensor[:,0,:] = torch.LongTensor(current_token)
        for i in range(max_len-1):
            print(final_output[:,:(i+1)].permute(1,0).shape)
            # print(current_token.shape)
            input_tensor = torch.LongTensor(current_token).unsqueeze(0)
            print(input_tensor.shape)
            input_tensor = self.embedding(input_tensor) * math.sqrt(self.d_model)
            input_tensor = self.pos_encoder(input_tensor)



            # input_tensor = input_tensor.permute(1, 0, 2)
            print(f"input ten {input_tensor.shape} mem {memory.shape}")
            # Forward pass through the decoder
            output = self.transformer_decoder(input_tensor, memory)
            output = self.fc_out(output)
            print(output.shape)
            outputs_tensor[:,i+1,:] = output
            max = torch.argmax(output,dim = 2)
            print(max)
            final_output[:,i+1] = torch.LongTensor(max).cpu()
            # break
        return final_output,outputs_tensor


if __name__ == "__main__":
    decoder = TransformerDecoder( vocab_size = 120, d_model = 512,        
        nhead = 2, 
        num_layers = 3, 
        dim_feedforward=2048, 
        dropout=0.2)
    x = torch.rand(2,512) 
    cap_tok = torch.rand(2,90) 
    decoder(x,cap_tok.long())
    decoder.predict(x)
