
import torch
from actor_critic import actor_critic

worker = actor_critic(
    model_dim = 64,
    mlp_dim = 128,
    heads = 4,
    enc_layers = 2,
    dec_layers = 2,
    max_sequence = 100,
    n_players = 6,
    gamma = 0.99,
    n_actions = 14
    )

with torch.autograd.set_detect_anomaly(True):
    loss  = worker.play_hand()
    print(loss)
    loss[0].backward(retain_graph=True)


