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

skill = np.zeros(64).astype(np.float32)
#skill = np.random.uniform(0,1,64).astype(np.float32)

expert_dataset = f_workspace.gather_trajectories(skill, 10000)
train_expert_dataset, test_expert_dataset = dataset_random_split(expert_dataset)

batch_size = 32
epochs = 10
train_loader = torch.utils.data.DataLoader(dataset=train_expert_dataset, batch_size=batch_size, shuffle=True)

from agent.ddpg import Actor
#p_workspace.agent.actor = Actor('states', 24, 6, 50, 64).to('cuda') #Actor('states', 24, 6, 50, 64).to('cuda')

actor_opt = torch.optim.Adam(p_workspace.agent.actor.parameters(), lr = 0.001)

print(p_workspace.agent.behavior_cloning(train_loader, actor_opt, epochs))
p_workspace.eval(skill)
