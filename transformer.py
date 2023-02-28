# Main model

import torch.nn as nn
import torch
from encoder import encoder
from decoder import decoder
from model_components import critic_head

class mod_transformer(nn.Module):
    # tested and functional

    def __init__(self,
    model_dim,
    mlp_dim,
    attn_heads,
    sequence_length,
    enc_layers,
    memory_layers,
    mem_length,
    dec_layers,
    action_dim,
    ) -> None:
        super().__init__()

        self.encoder = encoder(
            layers=enc_layers,
            model_dim=model_dim,
            mlp_dim=mlp_dim,
            heads=attn_heads,
        )

        self.decoder = decoder(
            layers=dec_layers,
            model_dim= model_dim,
            mlp_dim=mlp_dim,
            heads=attn_heads,
            sequence_length=sequence_length,
            memory_layers=memory_layers,
            mem_length=mem_length
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
        
        self.seen = {0: False, 1: False}

    def forward(self, enc_input, dec_input, player, new_hand, games_per_run):
        if new_hand:
            self.seen = [{0: False, 1: False}] * len(dec_input)
            self.hands = [{}] * len(dec_input)
        
        enc = [[] for _ in range(games_per_run)]
        for x in range(len(dec_input)):
            if self.seen[x][player[x]]:
                enc[x] = self.hands[x][player[x]]
        
            else:
                self.seen[x][player[x]] = True
                enc[x] = self.encoder(enc_input[x])
                self.hands[x][player[x]] = enc[x]
        enc = torch.stack(enc)
        dec = self.decoder(dec_input, enc) #expects a list of tensors as dec_input, enc_input
        policy_logits = self.actor(dec)
        value = self.critic(dec)
        return policy_logits, value