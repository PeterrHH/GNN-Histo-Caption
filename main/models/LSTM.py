
import torch.nn as nn
import torch


class LSTMDecoder(nn.Module):
    def __init__(self,vocab_size, embed_size, batch_size, hidden_size, device = "cpu",dropout = 0.5, num_layers = 1):
        super().__init__()
        assert dropout <= 1
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.device = device
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device
        self.dropout = nn.Dropout(p = self.dropout)
        self.word_embedding = nn.Embedding( self.vocab_size , self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first = True) # tensor shape (batch, seq, feature)  when batch_first = True
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self):
        return ( torch.zeros( self.num_layers , self.batch_size , self.hidden_size  ).to(self.device),
        torch.zeros( self.num_layers , self.batch_size , self.hidden_size  ).to(self.device) )

    def forward(self,features,captions):
        # decoded_output, _ = self.decoder(encoded_output)
        # return decoded_output
        captions = captions[:,:-1]
        h_0, c_0 = self.init_hidden()
        print(f"Hidden shape is {h_0.shape}")
        print(f"Cell shape is {c_0.shape}")
        embeds = self.word_embedding( captions)
        print(f"word embedding shape {embeds.shape} and features shape is {features.shape}")
        inputs = torch.cat( ( features, embeds ) , dim =1  ) 
        embeddings,_ = self.lstm(inputs,(h_0,c_0))
        outputs = self.linear(self.dropout(embeddings))
        return outputs

    # def init_hidden(self):
    #     h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
    #     c_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
    #     return h_0, c_0

    def predict(self, inputs, max_len=50):        
        final_output = []
        batch_size = inputs.shape[0]         
        hidden = self.init_hidden(batch_size) 
    
        while True:
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.fc(lstm_out) 
            print 
            outputs = outputs.squeeze(1) 
            _, max_idx = torch.max(outputs, dim=1) 
            final_output.append(max_idx.cpu().numpy()[0].item())             
            if (max_idx == 1 or len(final_output) >=20 ):
                break
            
            inputs = self.word_embedding(max_idx) 
            inputs = inputs.unsqueeze(1)             
        return final_output  

if __name__ == "__main__":
    import torch

    # Define hyperparameters
    vocab_size = 1000
    input_size = 128
    batch_size = 32
    hidden_size = 256
    seq_length = 20

    # Create a sample input
    sample_input = torch.randn(batch_size, seq_length, input_size)
    print("Input shape:", sample_input.shape)
    # Initialize the model
    model = LSTMDecoder(vocab_size, input_size, batch_size, hidden_size, device="cpu")

    # Get the output from the forward pass
    output = model(sample_input)

    print("Output shape:", output.shape)
    print(output[0])