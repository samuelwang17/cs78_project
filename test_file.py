
import torch
from actor_critic import actor_critic

worker = actor_critic(
    model_dim = 64,
    mlp_dim = 128,
    heads = 4,
    enc_layers = 2,
    dec_layers = 2,
    max_sequence = 100,
    n_players = 2,
    gamma = 0.99,
    n_actions = 14,
    memory_layers=0,
    mem_length=0,
<<<<<<< HEAD
    games_per_run=30,
=======
>>>>>>> parent of c6c6add (games per run functionality)
    )

with torch.autograd.set_detect_anomaly(True):
    while True:
        loss = worker.play_hand()
        print(loss)
        loss[0].backward(retain_graph=True)


