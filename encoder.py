# encoder layer and encoder itself

import torch
import torch.nn as nn
from model_components import self_attention, mlp, gru

class encoder_layer(nn.Module):
    # transformer layer
    # not masked, no cross attention, no memory, for encoder
    def __init__(self,
    embed_dim,
    mlp_dim,
    attention_heads,
    ) -> None:
        super().__init__()

        self.mha = self_attention(
            embed_dimension=embed_dim,
            num_heads=attention_heads
        )

        self.mlp = mlp(
            embed_dim= embed_dim,
            internal_dim=mlp_dim
        )

        self.gate1 = gru(
            dim = embed_dim
        )
        self.gate2 = gru(
            dim = embed_dim
        )

        self.ln1 = nn.LayerNorm(embed_dim)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(.1)
    
    def forward(self, x):
        y = self.ln1(x)
        y = self.mha(y)
        y = self.dropout(y)
        x = self.gate1(x,self.activation(y))
        y = self.ln1(x)
        y = self.mlp(y)
        x = self.gate2(x, self.activation(y))
        
        return x


class encoder(nn.Module):
    def __init__(self,
    layers,
    model_dim,
    mlp_dim,
    heads,
    ) -> None:
        super().__init__()
        
        # no inductive biases on encoder here
        self.block = nn.Sequential()
        for x in range(layers):
            self.block.append(encoder_layer(
                embed_dim = model_dim,
                mlp_dim = mlp_dim,
                attention_heads = heads,
            ))
            
    def forward(self, x):
        return self.block(x)