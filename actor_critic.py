# total actor critic package, encapsulating a single worker/env pair

import torch.nn as nn
from transformer import mod_transformer as RLformer
from tokenizer import Tokenizer

class Agent(nn.Module):
    def __init__(self,
        model_dim,
        mlp_dim,
        attn_heads,
        sequence_length,
        enc_layers,
        dec_layers,
        action_dim,
    ) -> None:
        super().__init__()
        self.model = RLformer(
            model_dim = model_dim,
            mlp_dim = mlp_dim,
            attn_heads = attn_heads,
            sequence_length = sequence_length,
            enc_layers = enc_layers,
            dec_layers = dec_layers,
            action_dim = action_dim,
        )

        self.tokenizer =  Tokenizer(model_dim=model_dim)
    
    def init_player(self, player, hand):
        # initialize this players hand and tokenize it, store it in buffer
        hand_tensor = Tokenizer(hand) #hand needs to be card observations -- list of length two of tensors
        self.register_buffer(f'hand_{player}', tensor= hand_tensor)

    def forward(self, player, obs_flat):
        #takes flattened inputs in list form, not tokenized
        enc_input = self.get_buffer(f'hand_{player}')
        dec_input = Tokenizer(obs_flat)
        policy, value = self.model(enc_input, dec_input)

        return policy, value