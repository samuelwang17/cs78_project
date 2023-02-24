# positional encoder, decoder layer, and full decoder

import torch
import torch.nn as nn
import math
from model_components import cross_attention, mlp, gru, decoder_mha


class positional_encoding(nn.Module):
    # tested and functional
    def __init__(self,
    model_dim,
    sequence_length
    ) -> None:
        super().__init__()

        position = torch.arange(sequence_length).unsqueeze(1)
        freq = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(1, sequence_length, model_dim)
        pe[0, :, 0::2] = torch.sin(position * freq)
        pe[0, :, 1::2] = torch.cos(position * freq)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]


class decoder_layer(nn.Module):
    # transformer layer
    # masked, cross attention, smeared key
    def __init__(self,
    embed_dim,
    mlp_dim,
    attention_heads,
    sequence_lenth
    ) -> None:
        super().__init__()

        self.mha = decoder_mha(
            model_dim=embed_dim,
            sequence_length=sequence_lenth,
            heads=attention_heads
        ) #smeared key masked self attention

        self.cross_mha = cross_attention(
            embed_dimension = embed_dim,
            num_heads = attention_heads,
        )

        self.mlp = mlp(
            embed_dim = embed_dim,
            internal_dim = mlp_dim
        )

        self.gate1 = gru(
            dim = embed_dim
        )
        self.gate2 = gru(
            dim = embed_dim
        )
        self.gate3 = gru(
            dim = embed_dim
        )

        self.ln = nn.LayerNorm(embed_dim)

        self.activation = nn.ReLU()
        self.ln1 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, enc):
        # masked self attention, smeared key
        y = self.ln1(x)
        y = self.mha(y)
        x = self.gate1(x,self.activation(y))

        # cross attention
        # consider output sequence length and 
        y = self.ln1(x)
        enc = self.ln1(enc)
        y = self.cross_mha(enc, x)
        x = self.gate2(x, self.activation(y))

        # position-wise multi layper perceptron
        y = self.ln1(x)
        y = self.mlp(y)
        x = self.gate2(x, self.activation(y))
        
        return x

class decoder(nn.Module):
    def __init__(self,
    layers,
    model_dim,
    mlp_dim,
    heads,
    sequence_length
    ) -> None:
        super().__init__()

        self.pe = positional_encoding(
            model_dim=model_dim, 
            sequence_length=sequence_length
            )

        self.block = []

        for x in range(layers):
            self.block.append(
                decoder_layer(
                    embed_dim = model_dim,
                    mlp_dim= mlp_dim,
                    attention_heads= heads,
                    sequence_lenth = sequence_length
                )
            )
        
    def forward(self, x, y):
        # y is input from encoder
        x = self.pe(x)
        for layer in self.block:
            x = layer(x,y)
            
        return x