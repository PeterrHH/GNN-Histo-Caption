from torch.nn import LSTM
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self,dropout = 0.5):
        super().__init__()
        assert dropout <= 1
        self.dropout_rate = dropout
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p = self.dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) # tensor shape (batch, seq, feature)  when batch_first = True
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self,feature, captions):
        # decoded_output, _ = self.decoder(encoded_output)
        # return decoded_output
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim = 0)
        hiddens, _ = self.lstm(embeddings) # features from the image
        outputs = self.linear(hiddens)
        return outputs