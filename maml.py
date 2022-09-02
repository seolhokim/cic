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

def clone_module(module, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)
    **Description**
    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().
    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.
    **Arguments**
    * **module** (Module) - Module to be cloned.
    **Return**
    * (Module) - The cloned module.
    **Example**
    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    # NOTE: This function might break in future versions of PyTorch.

    # TODO: This function might require that module.forward()
    #       was called in order to work properly, if forward() instanciates
    #       new variables.
    # NOTE: This can probably be implemented more cleanly with
    #       clone = recursive_shallow_copy(model)
    #       clone._apply(lambda t: t.clone())

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone

import torch.nn as nn
import utils

def inner_loop(agent, train_loader, test_loader, optimizer, epochs):
    metrics = dict()
    criterion = nn.MSELoss()
    agent.actor.train()
    total_train_loss = 0
    total_test_loss = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = torch.tensor(data, dtype=torch.float).to(agent.device), torch.tensor(target, dtype=torch.float).to(agent.device)
            stddev = utils.schedule(agent.stddev_schedule, 0) # step = 0  change
            dist = agent.actor(data, stddev)

            #action = dist.rsample()
            action = dist.sample(clip=agent.stddev_clip)

            loss = criterion(action, target)
            total_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            #optimizer.step()

    for data, target in test_loader:
        '''
        #why it works? -> rsample of TruncateNormal works.
        data, target = data.to(agent.device), target.to(agent.device)
        dist = agent.actor(torch.tensor(data, dtype=torch.float), 0.1)
        action = dist.rsample()
        '''
        data, target = torch.tensor(data, dtype=torch.float).to(agent.device), torch.tensor(target, dtype=torch.float).to(agent.device)
        stddev = utils.schedule(agent.stddev_schedule, 0) # step = 0  change
        dist = agent.actor(data, stddev)

        #action = dist.rsample()
        action = dist.sample(clip=agent.stddev_clip)
        loss = criterion(action, torch.tensor(target, dtype = torch.float))
        total_test_loss += loss
    # update actor
    metrics['bc_train_loss'] = (total_train_loss/epochs).item()

    return total_test_loss, metrics

epochs = 1
outer_loop_epochs = 5
inner_loop_epochs = 3
actor_opt = torch.optim.Adam(p_workspace.agent.actor.parameters(), lr = 0.001)

for _ in range(outer_loop_epochs):
    outer_loss = 0
    p_workspace.agent.actor.zero_grad()
    for i in range(inner_loop_epochs):
        skill = np.random.uniform(0,1,64).astype(np.float32)
        expert_dataset = f_workspace.gather_trajectories(skill, batch_size_per_skill)
        train_expert_dataset, test_expert_dataset = dataset_random_split(expert_dataset)
        train_expert_dataset, test_expert_dataset = dataset_random_split(expert_dataset)
        batch_size = 32
        epochs = 10
        train_loader = torch.utils.data.DataLoader(dataset=train_expert_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_expert_dataset, batch_size=len(test_expert_dataset), shuffle=True)


        temp_agent = clone_module(p_workspace.agent)
        temp_actor_opt = torch.optim.Adam(temp_agent.actor.parameters(), lr = 0.001)

        loss, metrics = inner_loop(temp_agent, train_loader, test_loader, temp_actor_opt, epochs)
        outer_loss += loss
        print(metrics)
    for p in p_workspace.agent.actor.parameters():
        p.grad.data.mul_(1.0 / inner_loop_epochs)
    outer_loss.backward()
    actor_opt.step()
    
torch.save(p_workspace.agent.actor.state_dict(), 'actor')

cfg = hydra.compose(config_name='maml_test.yaml', overrides=[])
maml_workspace = fW(cfg)
maml_workspace.agent.actor.load_state_dict(torch.load("actor"))
maml_workspace.train()