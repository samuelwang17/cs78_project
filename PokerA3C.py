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
    def __init__(self, global_actor_critic, global_ep_idx, optimizer, player_params, global_actor_ema, global_critic_ema, global_loss_ema):
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
        self.N_GAMES = 100000
        self.global_actor_ema = global_actor_ema
        self.global_critic_ema = global_critic_ema
        self.global_loss_ema = global_loss_ema


    def run(self):
        while self.episode_idx.value < self.N_GAMES:
            loss, actor_loss, critic_loss = self.local_actor_critic.play_hand()
            self.optimizer.zero_grad() # zero gradient on the master copy
            with torch.autograd.set_detect_anomaly(True):
                loss.backward(retain_graph=True)
            for local_param, global_param in zip(
                    self.local_actor_critic.agent.model.parameters(),
                    self.global_actor_critic.parameters()):
                global_param._grad = local_param.grad
            self.optimizer.step()
            self.local_actor_critic.agent.model.parameters = self.global_actor_critic.parameters
            self.local_actor_critic.clear_memory()
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            with self.global_loss_ema.get_lock():
                self.global_loss_ema.value = self.global_loss_ema.value * .999 + loss * .001 if self.global_loss_ema.value != 0 else loss
            with self.global_actor_ema.get_lock():
                self.global_actor_ema.value = self.global_actor_ema.value * .999 + actor_loss * .001 if self.global_actor_ema.value != 0 else actor_loss
            with self.global_critic_ema.get_lock():
                self.global_critic_ema.value = self.global_critic_ema.value * .999 + critic_loss * .001 if self.global_critic_ema.value != 0 else critic_loss
            if self.episode_idx.value % 100 == 0:
                print(f'{self.name} episode: {self.episode_idx.value}, loss: {self.global_loss_ema.value // 1}, actor loss: {self.global_actor_ema.value}, critic_loss: {self.global_critic_ema.value // 1}')


if __name__ == '__main__':
    torch.manual_seed(0)
    N_GAMES = 100
    actor_count = 12
    # actor parameters
    max_sequence = 100
    n_players = 2
    gamma = .7
    n_actions = 6
    # model parameters
    model_dim = 32
    mlp_dim = 64
    attn_heads = 4
    sequence_length = 50
    enc_layers = 3
    dec_layers = 8
    action_dim = 7
    learning_rate = .0001
    player_params = [model_dim, mlp_dim, attn_heads, enc_layers, dec_layers, sequence_length, n_players, learning_rate, action_dim]
    model_params = [model_dim, mlp_dim, attn_heads, sequence_length, enc_layers, dec_layers, action_dim]
    # create poker environment
    # initialize global model

    with open("hand_replays.txt", 'w') as file:
        file.writelines("Hand History \n")

    global_model = RLformer(* model_params)
    global_model.share_memory()
    print('Parameter_count: ', sum(p.numel() for p in global_model.parameters() if p.requires_grad))
    optimizer = SharedAdam(global_model.parameters(), lr=learning_rate)
    global_ep = mp.Value('i', 0)
    global_loss_ema = mp.Value('d', 0)
    global_actor_ema = mp.Value('d', 0)
    global_critic_ema = mp.Value('d', 0)
    # initialize players
    players = []
    for i in range(actor_count):
        player = Player(global_model, global_ep, optimizer, player_params, global_actor_ema, global_critic_ema, global_loss_ema)
        players.append(player)
    [player.start() for player in players]
    [player.join() for player in players]




