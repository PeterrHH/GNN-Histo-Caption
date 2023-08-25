from torch.nn import LSTM
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self,vocab_size, embed_size, dropout = 0.5, num_layers = 3):
        super().__init__()
        assert dropout <= 1
        self.dropout_rate = dropout
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p = self.dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) # tensor shape (batch, seq, feature)  when batch_first = True
        self.linear = nn.Linear(hidden_size, vocab_size)


    def forward(self,feature, captions):
        # decoded_output, _ = self.decoder(encoded_output)
        # return decoded_output
        # embeddings = self.dropout(self.embed(captions))
        h_0, c_0 = self.init_hidden(embeddings.size(0))
        # embeddings = torch.cat((features.unsqueeze(0),embeddings),dim = 0)
        hiddens, _ = self.lstm(embeddings, (h_0, c_0))
        outputs = self.linear(hiddens)
        return outputs

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h_0, c_0