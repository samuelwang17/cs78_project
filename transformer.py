# Main model

import torch.nn as nn
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
        
        # nn.Sequential(
        #     nn.Linear(model_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, 1)
        # )

        self.hands = {}
        self.seen = {0: False, 1: False}

    def forward(self, enc_input, dec_input, player, new_hand):
        if new_hand:
            self.seen = {0: False, 1: False}
        if self.seen[player]: 
            enc = self.hands[player]   
        else:
            self.seen[player] = True
            enc = self.encoder(enc_input)
            self.hands[player] = enc
        dec = self.decoder(dec_input, enc)
        policy_logits = self.actor(dec)
        value = self.critic(dec)
        return policy_logits, value