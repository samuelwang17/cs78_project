# Main model

import torch.nn as nn
import torch
from decoder import decoder
from model_components import critic_head

class mod_transformer(nn.Module):
    # tested and functional

    def __init__(self,
    model_dim,
    mlp_dim,
    attn_heads,
    sequence_length,
    dec_layers,
    action_dim
    ) -> None:
        super().__init__()
        self.decoder = decoder(
            layers=dec_layers,
            model_dim= model_dim,
            mlp_dim=mlp_dim,
            heads=attn_heads,
            sequence_length=sequence_length
        )

        self.actor = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, action_dim)
        ) # actor returns logits, softmax handled at higher level

        self.critic = critic_head(
            value_points=5,
            model_dim=model_dim,
            critic_dim=model_dim*2
        )

        self.hands = {}
        self.seen = {0: False, 1: False}

    def forward(self, input):
        dec = self.decoder(input)
        policy_logits = self.actor(dec)
        value = self.critic(dec)
        return policy_logits, value