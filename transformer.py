# Main model

import torch.nn as nn
from encoder import encoder
from decoder import decoder


class mod_transformer(nn.Module):
    # tested and functional

    def __init__(self,
    model_dim,
    mlp_dim,
    attn_heads,
    sequence_length,
    enc_layers,
    dec_layers,
    action_dim
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
        )

        self.actor = nn.Sequential(
            nn.Linear(model_dim, action_dim),
            nn.ReLU(),
        ) # actor returns logits, softmax handled at higher level

        self.critic = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1)
        )


    def forward(self, enc_input, dec_input):

        enc = self.encoder(enc_input)
        dec = self.decoder(dec_input, enc)
        policy_logits = self.actor(dec)
        value = self.critic(dec)
        return policy_logits, value