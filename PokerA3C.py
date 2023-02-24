import math

import torch
import torch.optim as optim
import os
from pokerenv import poker_env
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from transformer import *

'https://github.com/MorvanZhou/pytorch-A3C/blob/master/discrete_A3C.py'


class Player(mp.Process):
    def __init__(self, global_actor_critic, global_ep_idx, optimizer, player_params):
        super(Agent, self).__init__()
        self.local_actor_critic = actor_critic(player_params)
        self.global_actor_critic = global_actor_critic
        self.episode_idx = global_ep_idx
        self.optimizer = optimizer

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done = False
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                loss = self.local_actor_critic.calc_loss(done)
                self.optimizer.zero_grad()
                loss.backward()
                for local_param, global_param in zip(
                        self.local_actor_critic.parameters(),
                        self.global_actor_critic.parameters()):
                    global_param._grad = local_param.grad
                self.optimizer.step()
                self.local_actor_critic.load_state_dict(
                        self.global_actor_critic.state_dict())
                self.local_actor_critic.clear_memory()
            t_step += 1
        with self.episode_idx.get_lock():
            self.episode_idx.value += 1
        print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)


if __name__ == '__main__':
    torch.manual_seed(0)
    N_GAMES = 1000000000
    actor_count = 64
    # actor parameters
    max_sequence = 200
    n_players = 2
    gamma = .8
    n_actions = 10
    # model parameters
    model_dim = 0
    mlp_dim = 0
    attn_heads = 0
    sequence_length = 0
    enc_layers = 0
    dec_layers = 0
    action_dim = 0
    learning_rate = 0
    player_params = [max_sequence, n_players, gamma, n_actions]
    model_params = [model_dim, mlp_dim, attn_heads, sequence_length, enc_layers, dec_layers, action_dim]
    # create poker environment
    env = poker_env()
    # initialize global model
    global_model = RLformer(* model_params)
    global_model.share_memory()
    optimizer = SharedAdam(global_model.parameters(), lr=learning_rate)
    optimizer.share_memory()
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    # initialize players
    players = []
    for i in range(actor_count):
        player = Player(global_model, global_ep, optimizer, player_params)
        players.push(player)
    [player.start() for player in players]
    [player.join() for player in players]




