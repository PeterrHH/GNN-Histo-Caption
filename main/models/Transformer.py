import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import numpy as np
import math
#(self,vocabs,embed_size, batch_size, hidden_size,bi_direction, device = "cpu",dropout = 0.5, num_layers = 1)
class TransCapDecoder(nn.Module):
    def __init__(self, vocabs, embed_size, dim_feedforward, device = "cpu",dropout = 0.5, num_layers = 3, nhead = 8):
        super().__init__()
        self.embed_size = embed_size
        self.vocabs = vocabs
        self.vocab_size = len(self.vocabs)
        self.max_seq_length = 70
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = device
        self.start_token = self.vocabs.word2idx['<start>'] # Default
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.positional_encoding = PositionalEncoding(self.embed_size, self.max_seq_length).to(self.device)
        # self.feature_projection = nn.Linear(self.embed_size, self.embed_size).to(self.device)
        self.decoder_layer = TransformerDecoderLayer(self.embed_size, self.nhead, self.dim_feedforward,batch_first = True).to(self.device)
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, self.num_layers).to(self.device)
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size).to(self.device)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout_layer = nn.Dropout(p = self.dropout)

    def forward(self, features, captions,mask = None):
        # features = self.feature_projection(features)
        seq_len = captions.size(1)
        # features = features.unsqueeze(1).repeat(1, seq_len, 1)  # Add sequence dimension
        features = features.unsqueeze(1)
        captions = self.embedding(captions)
        captions = self.positional_encoding(captions)
        print(f"caption pos encoding {captions.shape}")
       #print(f"pos encoding shape {captions.shape} feature sshape {features.shape}")
        # captions = captions.permute(1,0,2)
        # features = features.permute(1,0,2)
        print(f"features shape {features.shape} cap {captions.shape}")
        tgt_mask = self.generate_tgt_mask(self.max_seq_length).to(self.device)
        #print(f"tgt {tgt_mask} ")
        decoder_output = self.transformer_decoder(captions, features,tgt_key_padding_mask = mask,tgt_mask = tgt_mask)
        #decoder_output = self.layer_norm(decoder_output)
        output = self.dropout_layer(self.fc_out(decoder_output))
        #print(f"decoder_output s {decoder_output.shape} output s {output.shape} feature {features.shape} cap {captions.shape}")
        return output

    def predict(self, features, max_length, beam_size=1):
        if beam_size == 1:
            return self._greedy_decode(features, max_length)
        else:
            return self._beam_search_decode(features, max_length, beam_size)
    
    def generate_tgt_mask(self,seq_length):
        """
        Generates a square causal mask for the target sequence.

        Args:
        seq_length (int): The length of the target sequence (caption).

        Returns:
        torch.Tensor: A square causal mask of shape (seq_length, seq_length).
        """
        mask = torch.triu(torch.ones(seq_length, seq_length) * float('-inf'), diagonal=1)
        return mask


    def _greedy_decode(self, features, max_length):
        batch_size = features.size(0)
        features = features.unsqueeze(1)
        #features = features.unsqueeze(1).repeat(1,max_length,1)  # Add sequence dimension
        # features = features.permute(1,0,2)
        #sprint(f"inference feature {features.shape}")
        input_ids = torch.full((batch_size, 1), fill_value=self.start_token , dtype=torch.int64).to(self.device)
        #print(f"input id shatart {input_ids}")
        logits_tensor = torch.zeros((batch_size, max_length, self.vocab_size)).to(self.device)
        final_output = torch.zeros(batch_size, max_length, dtype=torch.int64).to(self.device)
        final_output[:,0] = self.start_token
        for i in range(max_length - 1):
            #print(f"input ID at the start {input_ids.shape}")
            input_embeddings = self.embedding(input_ids)
            tgt_mask = self.generate_tgt_mask(i+1).to(self.device)
            # print(f"i = {i} mask = {tgt_mask}")
            input_embeddings = self.positional_encoding(input_embeddings)
            #print(f"inference input emb shape {input_embeddings.shape} feat {features.shape}")
            decoder_output = self.transformer_decoder(input_embeddings, features,tgt_mask = tgt_mask)

            #print(f"decoder_output {decoder_output.shape}")
            next_token_logits = self.dropout_layer(self.fc_out(decoder_output[:,-1,:]))
            #print(f"next logit {next_token_logits.shape}")
            logits_tensor[:,i,:] = next_token_logits
            next_token = torch.argmax(next_token_logits,dim = 1)
            # next_token = next_token_logits.argmax(dim=2, keepdim=True)
            #print(f"next logit {next_token_logits.shape} next tok {next_token.shape} is {next_token}")

            #print(f"i = {i} next tok {next_token.shape}")
            input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)
            #print(f"------")
            #print(f"at i {i} input_ids is {input_ids}")
            #input_ids = next_token[-1,:,:]

            # final_output[:,i+1] = input_ids.squeeze(1)

        print(f"------Example output ten------")
        # print(final_output[0])
        return input_ids,logits_tensor

    def _beam_search_decode(self, features, max_length, beam_size):
        # Implementation of beam search decoding
        pass

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):

        self.encoding = self.encoding.to(x.device) 
        # print(f"encoding x shape {x.shape}")
        # print(x + self.encoding[:, :x.size(1)])
        return x + self.encoding[:, :x.size(1)]
    
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000, dropout=0.1, ):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x.permute(1,0,2)
#         self.pe = self.pe.to(x.device)
#         print(f"x shape is {x.shape}")
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

if __name__ == "__main__":
    # Parameters
    #vocab_size = 128
    feature_size = 512  # Feature size from CNN
    d_model = 512
    nhead = 8
    num_decoder_layers = 6
    dim_feedforward = 2048
    max_seq_length = 70
    SOS_TOKEN = 0  # Assuming 0 is the index of the start-of-sentence token

    from Vocabulary import Vocabulary
    import pickle
    with open("new_vocab_bladderreport.pkl", 'rb') as file:
        vocabs = pickle.load(file)
    print(len(vocabs))
    model =  TransCapDecoder(
        vocabs = vocabs,
        embed_size =  feature_size,
        nhead = nhead, 
        num_layers = 3, 
        dim_feedforward= dim_feedforward, 
        dropout= 0.4,
        device = "cpu",
    )
    print(model)
    # Example inputs
    features = torch.rand(2,512)
    cap = torch.rand(2,70)
    # Define the integer range you want (e.g., [0, 10))
    int_range_low = 0
    int_range_high = 10

    # Scale and convert to integers
    captions = (cap * (int_range_high - int_range_low) + int_range_low).to(torch.int32)
    # Forward pass
    output = model(features, captions)
    print(f"output is {output.shape}")
    # Inference
    generated_captions,logit = model.predict(features, max_length=max_seq_length, beam_size=1)

    print(f"generated_captions {generated_captions}, logit {logit.shape}")