import math

import torch
import torch.optim as optim
import os
from pokerenv import poker_env
from actor_critic import *
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from transformer import *

'inspired from https://github.com/MorvanZhou/pytorch-A3C/blob/master/discrete_A3C.py'


class Player(mp.Process):
    def __init__(self, global_actor_critic, global_ep_idx, optimizer, player_params):
        super(Player, self).__init__()
        self.local_actor_critic = actor_critic(
            model_dim=player_params[0],
            mlp_dim=player_params[1],
            heads=player_params[2],
            enc_layers=player_params[3],
            dec_layers=player_params[4],
            max_sequence=player_params[5],
            n_players=player_params[6],
            gamma=player_params[7],
            n_actions=player_params[8]
        )
        self.global_actor_critic = global_actor_critic
        self.episode_idx = global_ep_idx
        self.optimizer = optimizer
        self.N_GAMES = 1000

    def run(self):
        t_step = 1
        while self.episode_idx.value < self.N_GAMES:
            done = False
            score = 0
            while not done:
                loss = self.local_actor_critic.play_hand()
                self.optimizer.zero_grad() # zero gradient on the master copy
                print(loss)
                loss.backward()
                for local_param, global_param in zip(
                        self.local_actor_critic.agent.model.parameters(),
                        self.global_actor_critic.parameters()):
                    global_param._grad = local_param.grad
                self.optimizer.step()
                self.local_actor_critic.agent.model.load_state_dict(
                        self.global_actor_critic.state_dict())
                self.local_actor_critic.clear_memory() # unsure what clear memory is -- zero grad? -- we don't want it to delete its sequence memory. I implemented the former
            t_step += 1
        with self.episode_idx.get_lock():
            self.episode_idx.value += 1
        print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)


if __name__ == '__main__':
    torch.manual_seed(0)
    N_GAMES = 100
    actor_count = 4
    # actor parameters
    max_sequence = 100
    n_players = 6
    gamma = .98
    n_actions = 14
    # model parameters
    model_dim = 64
    mlp_dim = 128
    attn_heads = 4
    sequence_length = 100
    enc_layers = 3
    dec_layers = 4
    action_dim = 14
    learning_rate = .0001
    player_params = [model_dim, mlp_dim, attn_heads, enc_layers, dec_layers, sequence_length, n_players, learning_rate, action_dim]
    model_params = [model_dim, mlp_dim, attn_heads, sequence_length, enc_layers, dec_layers, action_dim]
    # create poker environment
    # initialize global model
    global_model = RLformer(* model_params)
    global_model.share_memory()
    print('Parameter_count: ', sum(p.numel() for p in global_model.parameters() if p.requires_grad))
    optimizer = SharedAdam(global_model.parameters(), lr=learning_rate)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    # initialize players
    players = []
    for i in range(actor_count):
        player = Player(global_model, global_ep, optimizer, player_params)
        players.append(player)
    [player.start() for player in players]
    [player.join() for player in players]




