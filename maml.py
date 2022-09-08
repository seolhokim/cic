batch_size_per_skill = 3000
num_pretraining_skill = 10

from finetune import Workspace as fW
from pretrain import Workspace as pW
import os
import hydra
from omegaconf import OmegaConf
import numpy as np

import pathlib
import hydra
import torch
from utils import dataset_random_split
hydra.core.global_hydra.GlobalHydra.instance().clear()
config_dir = pathlib.Path('./')
hydra.initialize(config_path=config_dir)

cfg = hydra.compose(config_name='finetune.yaml', overrides=[])
f_workspace = fW(cfg)

cfg = hydra.compose(config_name='pretrain.yaml', overrides=[])
p_workspace = pW(cfg)

import numpy as np
import torch.nn as nn
import utils
class MAML:
    def __init__(self, actor, f_workspace, p_workspace):
        self.agent = actor
        self.f_workspace = f_workspace
        self.p_workspace = p_workspace
        
        self.batch_size = 32

        self.outer_loop_epochs = 1000
        self.inner_loop_epochs = 10
        self.inner_lr = 0.05
        self.outer_lr = 0.001
        self.bc_epochs = 10
        self.skill_num = 64

        self.criterion = nn.MSELoss()
        self.agent.train()
        
        self.actor_weights = list(self.agent.parameters())
        self.actor_opt = torch.optim.Adam(self.actor_weights, lr = self.outer_lr)
    def train(self):
        for outer_epoch in range(self.outer_loop_epochs):
            print("outer_loop_epochs :", outer_epoch)
            outer_loss = 0
            self.agent.zero_grad()
            for i in range(self.inner_loop_epochs):
                skill = np.random.uniform(0,1,self.skill_num).astype(np.float32)
                expert_dataset = self.f_workspace.gather_trajectories(skill, batch_size_per_skill)
                train_expert_dataset, test_expert_dataset = dataset_random_split(expert_dataset)
                train_loader = torch.utils.data.DataLoader(dataset=train_expert_dataset, batch_size=self.batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(dataset=test_expert_dataset, batch_size=len(test_expert_dataset), shuffle=True)

                #loss, metrics = self.inner_loop(train_loader, test_loader)
                loss, metrics = self.inner_loop(train_loader, test_loader)
                outer_loss += loss
                print(metrics)
            meta_grads = torch.autograd.grad(outer_loss,self.actor_weights)

            for w, g in zip(self.actor_weights, meta_grads):
                w.grad = g
            for p in self.actor_weights:
                p.grad.data.mul_(1.0 / self.inner_loop_epochs)
            self.actor_opt.step()    
    def inner_loop_test(self, train_loader, test_loader):
        metrics = dict()
        total_train_loss = 0
        for epoch in range(self.bc_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.type(torch.float).to(self.p_workspace.device), target.type(torch.float).to(self.p_workspace.device)
                stddev = utils.schedule(self.p_workspace.agent.stddev_schedule, 0) # step = 0  change
                
                #dist = self.agent(data, stddev, temp_weights)
                dist = self.agent(data, stddev)

                action = dist.sample(clip=self.p_workspace.agent.stddev_clip)

                loss = self.criterion(action, target)
                self.agent.zero_grad()
                loss.backward()
                self.actor_inner_loop_test_opt.step()
                total_train_loss += loss
        metrics['bc_train_loss'] = (total_train_loss/self.bc_epochs).item()
        return total_train_loss, metrics
    def inner_loop(self, train_loader, test_loader):
        metrics = dict()
        total_train_loss = 0
        total_test_loss = 0
        temp_weights = [w.clone() for w in self.actor_weights]
        # temp_weights = self.actor_weights works 5
        # actor_inner_loop_test_opt = torch.optim.Adam(temp_weights, lr = self.inner_lr) 5
        for epoch in range(self.bc_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.type(torch.float).to(self.p_workspace.device), target.type(torch.float).to(self.p_workspace.device)
                stddev = utils.schedule(self.p_workspace.agent.stddev_schedule, 0) # step = 0  change
                
                #dist = self.agent(data, stddev, temp_weights)
                #dist = self.agent(data, stddev, self.actor_weights)
                #dist = self.agent(data, stddev)
                #dist = self.agent(data, stddev, self.actor_weights)
                dist = self.agent(data, stddev, temp_weights) #works 5
                
                action = dist.sample(clip=self.p_workspace.agent.stddev_clip)

                loss = self.criterion(action, target)
                
                #grad = torch.autograd.grad(loss, temp_weights)
                #grad = torch.autograd.grad(loss, self.actor_weights)
                #grad = torch.autograd.grad(loss, self.agent.parameters())
                #grad = torch.autograd.grad(loss, self.actor_weights)
                grad = torch.autograd.grad(loss, temp_weights) #works 5

                #temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)] 
                #self.actor_weights = [w - self.inner_lr * g for w, g in zip(self.actor_weights, grad)] 
                #for w, g in zip(temp_weights, grad): #5 works
                #    w.grad = g #5works

                #actor_inner_loop_test_opt.step() 5 works
                #print(grad[0][0])
                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)] 
                total_train_loss += loss
                #self.actor_weights = [nn.Parameter(w.detach()) for w in temp_weights]
                
        for data, target in test_loader:
            data, target = data.type(torch.float).to(self.p_workspace.device), target.type(torch.float).to(self.p_workspace.device)
            stddev = utils.schedule(self.p_workspace.agent.stddev_schedule, 0) # step = 0  change
            
            #dist = self.agent(data, stddev, temp_weights)
            dist = self.agent(data, stddev, self.actor_weights)
            
            action = dist.sample(clip=self.p_workspace.agent.stddev_clip)
            loss = self.criterion(action, target)
            total_test_loss += loss

        metrics['bc_train_loss'] = (total_train_loss/self.bc_epochs).item()

        return total_test_loss, metrics
    
from agent.ddpg import Actor

obs_type, obs_dim, action_dim, feature_dim, hidden_dim = 'state', 88, 6, 1024,1024
actor = Actor(obs_type, obs_dim, action_dim, feature_dim, hidden_dim).to('cuda')

maml = MAML(actor, f_workspace, p_workspace)
maml.train()