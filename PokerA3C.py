import math

import torch
import torch.optim as optim
import os
from actor_critic import *
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from transformer import *

'inspired from https://github.com/MorvanZhou/pytorch-A3C/blob/master/discrete_A3C.py'


class Player(mp.Process):
    def __init__(self, global_actor_critic, global_ep_idx, optimizer, player_params, global_actor_ema, global_critic_ema, global_loss_ema, global_inference_ema, token_args):
        super(Player, self).__init__()
        self.local_actor_critic = actor_critic(
            model_dim=player_params[0],
            mlp_dim=player_params[1],
            heads=player_params[2],
            dec_layers=player_params[3],
            max_sequence=player_params[4],
            n_players=player_params[5],
            gamma=player_params[6],
            n_actions=player_params[7],
            token_args=token_args
        )
        self.global_actor_critic = global_actor_critic
        self.episode_idx = global_ep_idx
        self.optimizer = optimizer
        self.N_GAMES = 5000000
        self.global_actor_ema = global_actor_ema
        self.global_critic_ema = global_critic_ema
        self.global_loss_ema = global_loss_ema
        self.global_inference_ema = global_inference_ema


    def run(self):
        while self.episode_idx.value < self.N_GAMES:
            loss, actor_loss, critic_loss, time_dict = self.local_actor_critic.play_hand()
            
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            
            if self.episode_idx.value % 10 == 0:
                for local_param, global_param in zip(
                    self.local_actor_critic.agent.model.parameters(),
                    self.global_actor_critic.parameters()):
                    global_param._grad = local_param.grad
                    #print(local_param.grad)
                self.optimizer.step()
                self.optimizer.zero_grad() # zero gradient on the master copy
                self.local_actor_critic.agent.model.load_state_dict(self.global_actor_critic.state_dict())
                #self.local_actor_critic.agent.model.parameters = self.global_actor_critic.parameters
                self.local_actor_critic.clear_memory()
                
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            with self.global_loss_ema.get_lock():
                self.global_loss_ema.value = self.global_loss_ema.value * .995 + loss * .005 if self.global_loss_ema.value != 0. else loss
            with self.global_actor_ema.get_lock():
                self.global_actor_ema.value = self.global_actor_ema.value * .995 + actor_loss * .005 if self.global_actor_ema.value != 0. else actor_loss
            with self.global_critic_ema.get_lock():
                self.global_critic_ema.value = self.global_critic_ema.value * .995 + critic_loss * .005 if self.global_critic_ema.value != 0. else critic_loss
            with self.global_inference_ema.get_lock():
                self.global_inference_ema.value = self.global_inference_ema.value * .995 + time_dict['total'] * .005 if self.global_inference_ema.value != 0. else time_dict['total']
            if self.episode_idx.value % 100 == 0:
                print(f'episode: {self.episode_idx.value}, loss: {self.global_loss_ema.value}, actor loss: {self.global_actor_ema.value}, critic loss: {self.global_critic_ema.value}, inference time: {self.global_inference_ema.value}')
            if self.episode_idx.value % 20000 == 0:
                torch.save(global_model, 'PokerA3CModel_new')

if __name__ == '__main__':
    torch.manual_seed(0)
    N_GAMES = 100
    actor_count = 12
    # actor parameters
    max_sequence = 200
    n_players = 2
    gamma = .95
    n_actions = 6
    # model parameters
    model_dim = 128
    mlp_dim = 256
    attn_heads = 8
    sequence_length = 50
    dec_layers = 4
    action_dim = 6
    learning_rate = 5e-4
    weight_decay = 1e-5
    token_args = {'pot_edim': 32, 'pos_edim': 16, 'action_edim': 64, 'stack_edim': 16 , 'card_edim': model_dim, 'n_players': 2}
    player_params = [model_dim, mlp_dim, attn_heads, dec_layers, sequence_length, n_players, gamma, action_dim]
    model_params = [model_dim, mlp_dim, attn_heads, sequence_length, dec_layers, action_dim]
    # create poker environment
    # initialize global model

    with open("hand_replays.txt", 'w') as file:
        file.writelines("Hand History \n")

    global_model = mod_transformer(* model_params)
    global_model.share_memory()
    print('Parameter_count: ', sum(p.numel() for p in global_model.parameters() if p.requires_grad))
    optimizer = SharedAdam(global_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    global_ep = mp.Value('i', 0)
    global_loss_ema = mp.Value('d', 0)
    global_actor_ema = mp.Value('d', 0)
    global_critic_ema = mp.Value('d', 0)
    global_inference_ema = mp.Value('d', 0)
    # initialize players
    players = []
    for i in range(actor_count):
        player = Player(global_model, global_ep, optimizer, player_params, global_actor_ema, global_critic_ema, global_loss_ema, global_inference_ema, token_args)
        players.append(player)
    [player.start() for player in players]
    [player.join() for player in players]
    torch.save(global_model, 'PokerA3CModel_new')




 