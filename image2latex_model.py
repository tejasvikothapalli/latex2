import math
import torch
import torch.nn as nn
from resnet_custom import *

from transformer_util import *


SEQ_LENGTH = 150
TRANSFORMER_DIM = 512
TRANSFORMER_FC_DIM = 1024
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_LAYERS = 4
N_HEADS = 4
N_BEAMS = 1


class Image2LatexModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.index2word = kwargs.get('index2word', None)
        self.word2index = kwargs.get('word2index', None)
        self.pad_index = self.word2index['<PAD>']
        self.start_index = self.word2index['<START>']
        self.end_index = self.word2index['<END>']
        
        self.seq_length = kwargs.get('seq_length', SEQ_LENGTH)
        self.transformer_dim = kwargs.get('transformer_dim', TRANSFORMER_DIM)
        self.transformer_fc_dim = kwargs.get('transformer_fc_dim', TRANSFORMER_FC_DIM)
        self.transformer_dropout = kwargs.get('transformer_dropout', TRANSFORMER_DROPOUT)
        self.transformer_layers = kwargs.get('transformer_layers', TRANSFORMER_LAYERS)
        self.n_heads = kwargs.get('n_heads', N_HEADS)
        self.vocab_size = len(self.index2word)
        self.n_beams = kwargs.get('n_beams', N_BEAMS)
        self.use_transformer_encoder = kwargs.get('use_transformer_encoder', True)
        
        self.resnet = resnet18()
        self.resnet_projection = nn.Conv2d(in_channels=512, 
                                           out_channels=self.transformer_dim, 
                                           kernel_size=1,
                                           stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, self.seq_length))
        self.positional_encoding = PositionalEncoding(d_model=self.transformer_dim, 
                                                     max_len=self.seq_length)
        self.positional_encoding_2d = PositionalEncodingImage(d_model=self.transformer_dim,
                                                              max_h=100,
                                                              max_w=100)
        if self.use_transformer_encoder:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim,
                                                       nhead=self.n_heads,
                                                       dim_feedforward=self.transformer_fc_dim,
                                                       dropout=self.transformer_dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                             self.transformer_layers)
        
        self.embedding = nn.Embedding(self.vocab_size, self.transformer_dim)
        self.y_mask = generate_square_subsequent_mask(self.seq_length)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.transformer_dim,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.transformer_fc_dim,
                                                   dropout=self.transformer_dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer,
                                                         self.transformer_layers)
        self.fc = nn.Linear(self.transformer_dim, self.vocab_size)
        
        nn.init.kaiming_normal_(self.resnet_projection.weight, mode='fan_out', nonlinearity='relu')
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        self.fc.bias.data.zero_()
        nn.init.kaiming_uniform_(self.fc.weight, mode='fan_out', nonlinearity='relu')

    def encode(self, x):
        """
        Execution steps:
        1. resnet: (Any size) -> (seq_length, batch_size, transformer_dim)
        2. positional encoding
        3. transformer_encoder
        4. positional encoding
        """
        batch_size = x.shape[0]
        x = self.resnet(x) # (batch_size, 512, image_h // 16, image_w // 16)
        x = self.resnet_projection(x) # (batch, trans_dim, image_h // 16, image_w // 16)
        
        # x = self.avgpool(x) # (batch_size, transformer_dim, 1, seq_len)
        # x = x.squeeze(2) # (batch, trans_dim, seq_len)
        # x = x.permute((2, 0, 1)) # (seq_len, batch, trans_dim)
        
        x = self.positional_encoding_2d(x)
        x_transformer = x.view((batch_size, self.transformer_dim, -1)).permute((2, 0, 1)) # (h * w, batch, trans_dim)
        if self.use_transformer_encoder:
            x_transformer = self.transformer_encoder(x_transformer)
            # x = self.positional_encoding_2d(x) # want to apply positional encoding at the original shape again
        
        return x_transformer
        
    
    def decode(self, x, y):
        """
        x: output from encode
        y: one-hot sequence of target sentence (assumed to start with <START> token)
        
        Execution steps:
        1. Embedding Layer: (seq_length, batch_size, vocab_size) -> (seq_length, batch_size, t_dim)
        2. Positional encoding
        3. transformer_decoder
        4. FC layer: (seq, batch, t_dim) -> (seq, batch, vocab)
        """
        # y starts as (batch_size, seq_length)
        _, seq_length = y.shape
        y = self.embedding(y) * math.sqrt(self.transformer_dim) # (batch, seq_length, trans_dim)
        y = y.permute((1, 0, 2)) # (seq_length, batch, trans_dim)
        y = self.positional_encoding(y)
        y_mask = self.y_mask[:seq_length, :seq_length].type_as(x)
        y = self.transformer_decoder(tgt=y,
                                     memory=x,
                                     tgt_mask=y_mask)
        y = self.fc(y) # (seq_len, batch, vocab)
        return y

    def forward(self, x, y):
        """
        This method is used during training
        The sequence fed into the decoder is assumed to have <START> and <END> tokens
        Execution steps:
        1. Feed thru encoder
        2. Feed thru decoder
        """
        x = self.encode(x)
        output = self.decode(x, y)
        return output
    
    # @torch.jit.export
    def predict(self, x):
        """
        Execution Steps:
        1. Send x thru encoder
        2. Start beam search
            a. Create empty array of size (beam, seq_length + 1) filled with <PAD> token
            b. First tokens are <START> tokens
            c. Create 
        
        Beam search step process:
        Inputs: beams (beam, len_so_far), log_beam_probs: (beam)
        1. Run beams thru decoder (seq, beam, vocab)
        2. Extract last tokens: (beam, vocab)
        3. Take log prob of last tokens
        4. Unsqueeze log_beam_probs to (beam, 1), add to last tokens
        5. Flatten and take top beam entries
            a. torch.topk returns values and indices of top k elements
            b. indices // n_beams is which beam to append to
            c. indices % n_beams is which vocab_idx to use
        6. Use indices to construct new beams
        7. The values returned from topk are the new log_beam_probs
        """
        with torch.no_grad():
            # Assume x is of shape (1, 3, image_h, image_w)
            x = self.encode(x) # (seq, 1, trans_dim)
            x = x.expand(-1, self.n_beams, -1)
        
            beams_so_far = self.start_index * torch.ones((self.n_beams, 1), dtype=torch.long)
            log_beam_probs = -1000.0 * torch.ones((self.n_beams), dtype=torch.float)
            log_beam_probs[0] = 0.0
            
            softmax = nn.LogSoftmax(-1)
            
            for i in range(self.seq_length):
                # print(beams_so_far, log_beam_probs)
                out = self.decode(x, beams_so_far) # (seq, beam, vocab)
                log_out = softmax(out)
                log_last_tokens = log_out[-1] # (beam, vocab)
                # print(log_last_tokens)
                
                log_last_tokens += log_beam_probs.unsqueeze(1)
                # print(log_last_tokens)
                log_beam_probs, indices = torch.topk(log_last_tokens.flatten(), self.n_beams) # (beam, )
                beams_to_use = indices // self.vocab_size
                vocab_indices = indices % self.vocab_size
                # print(beams_to_use, vocab_indices)
                beams_to_add = torch.stack([beams_so_far[idx] for idx in beams_to_use])
                beams_so_far = torch.cat((beams_to_add, vocab_indices.unsqueeze(1)), 1)
            
            return beams_so_far
        
