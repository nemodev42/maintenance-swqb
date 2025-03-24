from typing import Optional

from matplotlib import pyplot as plt # for plotting
import numpy as np # for cpu based computation
import torch # for efficient (gpu) computation and automatic differentiation
from tqdm import tqdm # for progress bars
from tensordict import TensorDict, TensorDictBase # for handling dictionaries of tensors in a pytorch friendly way, e.g. for batched data
from torch import nn # for neural networks
import torch.optim as optim # for optimizers
import torch.nn.functional as F # for activation functions
from torch.utils.tensorboard import SummaryWriter # for logging to tensorboard


# TorchRL
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec # for defining the shape and type of data [Legacy]
from torchrl.data import Bounded, Composite, Unbounded # for defining the shape and type of data
from torchrl.envs import (
    CatTensors, # Concatenate tensors
    EnvBase, # Tensordict based env
    Transform, # Transform for envs
)
from torchrl.envs.transforms.transforms import _apply_to_composite # for applying a transform to a composite spec
from torchrl.envs.utils import check_env_specs, step_mdp # for checking env specs and stepping through an MDP


# This code is optimized to run on cuda (gpu) if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_default_device(device)


# Neural network
class LinearQAK(nn.Module,):
    def __init__(self, hyperparameters):
        n_observations = hyperparameters["N_CONDITION_STATES"]
        self.basis = hyperparameters["BASIS_DOMAIN"]
        super(LinearQAK, self).__init__()
        self.one_layer = nn.Linear(n_observations, hyperparameters["DEGREE_APPROXIMATION"])
    def forward(self, x):
        x= self.one_layer(x)
        
        # evaluate q(s, a) for all values of f_k(a)
        x = x @ self.basis
        
        return  x # return q(s, a)

class LinearQ(nn.Module,):
    def __init__(self, hyperparameters):
        n_observations = hyperparameters["N_CONDITION_STATES"]
        self.basis = hyperparameters["BASIS_DOMAIN"]

        super(LinearQ, self).__init__()
        self.one_layer = nn.Linear(n_observations, hyperparameters["MAX_REPAIR_CONSTRAINT"] + 1)
    def forward(self, x):
        x= self.one_layer(x)
        
        return  x # return q(s, a)


# Neural network
class DQAKN(nn.Module,):
    def __init__(self, hyperparameters):
        n_observations = hyperparameters["N_CONDITION_STATES"]
        self.basis = hyperparameters["BASIS_DOMAIN"]

        super(DQAKN, self).__init__()
        # hidden layers
        self.layer1 = nn.Linear(n_observations, hyperparameters["N_DEEP_NODES"])
        self.layer3 = nn.Linear(hyperparameters["N_DEEP_NODES"], hyperparameters["DEGREE_APPROXIMATION"])

    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = self.layer3(x)
        
        # evaluate q(s, a) for all values of f_k(a)
        x = x @ self.basis
        
        return  x # return q(s, a)


# Neural network
class DQN(nn.Module,):
    def __init__(self, hyperparameters):
        n_observations = hyperparameters["N_CONDITION_STATES"]
        n_actions = hyperparameters["MAX_REPAIR_CONSTRAINT"] + 1
        self.basis = hyperparameters["BASIS_DOMAIN"]

        super(DQN, self).__init__()
        # hidden layers
        self.layer1 = nn.Linear(n_observations, hyperparameters["N_DEEP_NODES"])
        # self.layer2 = nn.Linear(N_DEEP_NODES, N_DEEP_NODES)
        # hidden layers to basis weights
        # self.layer3 = nn.Linear(hyperparameters["N_DEEP_NODES"], hyperparameters["DEGREE_APPROXIMATION"])
        # For no function approximation
        self.direct_layer = nn.Linear(hyperparameters["N_DEEP_NODES"], n_actions)
        # for no hidden laters
        # self.one_layer = nn.Linear(n_observations, DEGREE_APPROXIMATION)
    def forward(self, x):
        # x = self.one_layer(x)
        x = F.tanh(self.layer1(x))
        # x = F.tanh(self.layer2(x))
        x = self.direct_layer(x)
        # x = self.layer3(x)
        
        # evaluate q(s, a) for all values of e_k(a)
        # x = x @ self.basis
        
        return  x # return q(s, a)


# trainer class
class MaitenanceDQBNTrainer:
    def __init__(self, hyperparameters, env, seed = 42, model=None, target_model=None,):
        # set the seed
        torch.manual_seed(seed)

        self.hyperparameters = hyperparameters

        self.reset_memory()

        self.env = env

        self.q_hat = model
        self.q_hat_prime = target_model

        self.basis = hyperparameters["BASIS_DOMAIN"]

        self.tb_writer = SummaryWriter()

        if model is not None:
            self.set_optimizer()
    
    def reset_memory(self):
        self.replay_memory = TensorDict({}, [self.hyperparameters["BUFFER_SIZE"]])
        self.memory_index = 0

    # add a single transition
    def add_memory(self, observation, action, next_observation, reward, done):
        global memory_index

        memory = TensorDict(
            {
                "observation": observation.clone(),
                "action": action,
                "next_observation": next_observation,
                "reward": reward,
                "done": done,
            }
        )

        self.replay_memory[memory_index] = memory

        memory_index = (memory_index + 1) % self.hyperparameters["BUFFER_SIZE"]

    # add a batch of transitions
    def batch_add_memory(self, batch_observation, batch_action, batch_next_observation, batch_reward, batch_done):
        input_batch_size = batch_observation.shape[0]

        wrapped_keys = (torch.arange(input_batch_size) + self.memory_index) % self.hyperparameters["BUFFER_SIZE"]

        self.replay_memory[wrapped_keys] = TensorDict(
            {
                "observation": batch_observation,
                "action": batch_action,
                "next_observation": batch_next_observation,
                "reward": batch_reward,
                "done": batch_done,
            }
        )

        self.memory_index = (self.memory_index + input_batch_size) % self.hyperparameters["BUFFER_SIZE"]

    # create models
    def create_linear_models(self):
        # create policy and target networks
        self.q_hat = LinearQ( self.hyperparameters).to(device)
        self.q_hat_prime = LinearQ( self.hyperparameters).to(device)

        # set target network weights to policy network weights
        self.q_hat_prime.load_state_dict((self.q_hat).state_dict())

        self.set_optimizer()

    def create_linear_AK_models(self):
        # create policy and target networks
        self.q_hat = LinearQAK( self.hyperparameters).to(device)
        self.q_hat_prime = LinearQAK( self.hyperparameters).to(device)

        # set target network weights to policy network weights
        self.q_hat_prime.load_state_dict((self.q_hat).state_dict())

        self.set_optimizer()

    def create_models(self):
        # create policy and target networks
        self.q_hat = DQN( self.hyperparameters).to(device)
        self.q_hat_prime = DQN( self.hyperparameters).to(device)

        # set target network weights to policy network weights
        self.q_hat_prime.load_state_dict((self.q_hat).state_dict())

        self.set_optimizer()

    def create_AK_models(self):
        # create policy and target networks
        self.q_hat = DQAKN( self.hyperparameters).to(device)
        self.q_hat_prime = DQAKN( self.hyperparameters).to(device)

        # set target network weights to policy network weights
        self.q_hat_prime.load_state_dict((self.q_hat).state_dict())

        self.set_optimizer()

    def set_optimizer(self):
        # Schedule learning rate
        self.steps_done = 0
        LR_schedule = torch.linspace(self.hyperparameters["LR"], 0, self.hyperparameters["EPOCHS"])
        def LR_scheduler():
            return LR_schedule[self.steps_done]

        # Create optimizer
        self.optimizer = optim.AdamW(self.q_hat.parameters(), lr=LR_scheduler(), amsgrad=True)
    
    # Select action from policy network
    # state - state to select action
    # entropy - chance to take random action
    def select_action(self, state, entropy=0.0):
        # get the batch size
        batch_size = state["observation"].shape[0]
        # create a tensor to store the actions
        actions = torch.zeros(batch_size, self.hyperparameters["N_COMPONENTS"], dtype=torch.int64)
        # generate random number for random actions
        sample = torch.rand(1, device=device)
        n_repair = None
        
        if sample < entropy:
            # take random action
            n_repair = torch.randint(0, self.hyperparameters["MAX_REPAIR_CONSTRAINT"] + 1, (batch_size,), device=device)
        else: 
            # take action from policy network
            with torch.no_grad(): # Do not calculate gradient.
                q_values = self.q_hat(state["observation"])
                n_repair = torch.argmax(q_values, dim=1)
        
        # n_repair is the number of components to repair
        # repair starting from compoennts with the highest (worst) condition
        
        conditions = state["conditions"]

        # sort components with the highest condition
        components_by_condition = torch.argsort(conditions, dim=1, descending=True)
        
        # Create a mask for the top-k indices
        topk_mask = torch.arange(self.hyperparameters["N_COMPONENTS"], device=device).expand(batch_size, self.hyperparameters["N_COMPONENTS"]) < n_repair.unsqueeze(1)

        # indicies of batches of components
        batch_indicies = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, self.hyperparameters["N_COMPONENTS"])[ topk_mask ]

        # indicies of components to be repaired
        topk_indices = components_by_condition[ topk_mask ]

        # set the actions of the components to be repaired
        actions[batch_indicies, topk_indices] = 1

        # return selected actions and number of components repaired in each action
        return actions, n_repair.squeeze()

    # optimize the policy network
    def optimize_model(self):
        # return if not enough memory for optimization batch
        if len(self.replay_memory) < self.hyperparameters["OPTIMIZATION_BATCH_SIZE"]:
            return
        
        # sample a batch of transitions
        sample_index = torch.randint(0, len(self.replay_memory), (self.hyperparameters["OPTIMIZATION_BATCH_SIZE"],))
        batch = self.replay_memory[sample_index]

        state_batch = batch["observation"]
        action_batch = batch["action"]
        next_state_batch = batch["next_observation"]
        reward_batch = batch["reward"]

        # see what policy network predicts on initial states
        # NOTE: Gradient Calculated with this operation!
        state_action_values = self.q_hat(state_batch.squeeze())[torch.arange(state_batch.shape[0]), action_batch]

        # see what target network predicts on next states
        next_state_values = torch.zeros_like(state_action_values)
        with torch.no_grad(): # do not calculate gradient on target network. (Target network is updated softly through tau)
            Q = self.q_hat_prime(next_state_batch)
            _, Q_best_index = torch.max(Q, dim=1)
            next_state_values = Q[torch.arange(state_action_values.shape[0]), Q_best_index]

        # calculate tempotal difference target
        target_state_action_values = (next_state_values * self.hyperparameters["GAMMA"]) + reward_batch

        # get loss
        criterion = nn.SmoothL1Loss(reduction="none")
        loss = criterion(state_action_values, target_state_action_values)
        loss = loss.mean()



        self.tb_writer.add_scalar("loss", loss.item(), self.steps_done)
        
        # optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.q_hat.parameters(), .1)
        self.optimizer.step()


    def train(self, verbose=True, test_observation=None):
        # self.tb_writer = SummaryWriter()
        self.tb_writer.add_text("hyperparameters", str(self.hyperparameters.to_dict()))
        
        # linear decay
        # exploration_scheduler = torch.linspace(self.hyperparameters["ENTROPY_START"], self.hyperparameters["ENTROPY_END"], self.hyperparameters["EPOCHS"])
        # exponential decay
        EPS_END = self.hyperparameters["ENTROPY_END"]
        EPS_START = self.hyperparameters["ENTROPY_START"]
        EPS_DECAY = self.hyperparameters["ENTROPY_DECAY"]
        exploration_scheduler = [ EPS_END + (EPS_START - EPS_END) * \
        torch.exp(-1. * t / EPS_DECAY) for t in range(self.hyperparameters["EPOCHS"])]

        test_q_cache = None
        if test_observation is not None:
            test_q_cache = torch.zeros(self.hyperparameters["EPOCHS"], self.hyperparameters["MAX_REPAIR_CONSTRAINT"] + 1)
            test_observation = torch.cat([test_observation.unsqueeze(0), test_observation.unsqueeze(0),], dim=0)
            # print(test_observation.shape)


        pbar = None
        if verbose:
            # for progress bar durring training
            pbar = tqdm(range(self.hyperparameters["EPOCHS"]))

        for episode in range(self.hyperparameters["N_EPISODES"]):
            # reset the enviroment tensordict
            td = self.env.reset(self.env.gen_params(batch_size=[self.hyperparameters["STEP_BATCH_SIZE"]], device=device))

            episode_reward = 0

            action_count = np.zeros((self.hyperparameters["STEP_BATCH_SIZE"], self.hyperparameters["EPISODE_LENGTH"]))

            for step in range(self.hyperparameters["EPISODE_LENGTH"]):
                if verbose:
                    pbar.update(1)

                # get the current state
                observation = td["observation"]

                # print(observation[0].sum())


                # select action
                actions, n_repair = self.select_action(td, entropy=exploration_scheduler[self.steps_done])

                if verbose:
                    # update progress bar
                    pbar.set_description(f"entropy: {exploration_scheduler[self.steps_done]:.2f}")

                # add n_repair to action count, accounting for batch size
                # for i in range(n_repair.shape[0]):
                #     action_count[n_repair[i].cpu().item()] += 1
                # vectorized
                
                action_count[:, step] = n_repair.cpu().numpy()

                # set the action in the tensordict
                td["action"] = actions

                # step the enviroment
                td = self.env.step(td)

                # move tensordict to next state
                td = td["next"]

                # get next observation
                next_observation = td["observation"]

                # get rewards
                rewards = td["reward"].squeeze()

                episode_reward += rewards.mean().cpu().item()

                # get done
                done = td["done"]

                # add batch transition to memory replay buffer
                self.batch_add_memory(observation, n_repair, next_observation, rewards, done)
                
                self.steps_done += 1

                # if buffer is full, optimize the policy network
                if self.steps_done * self.hyperparameters["STEP_BATCH_SIZE"] > self.hyperparameters["BUFFER_SIZE"]:
                    # optimize for each pass set in hyperparameters
                    for j in range(self.hyperparameters["OPTIMIZATION_PASSES"]):
                        self.optimize_model()

                    # update target network
                    target_net_state_dict = self.q_hat_prime.state_dict()
                    policy_net_state_dict = self.q_hat.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * self.hyperparameters["TAU"] + target_net_state_dict[key] * (1 - self.hyperparameters["TAU"]) # DONT FOTGET 1 - TAU, IT WILL TAKE A WHOLE DAY TO FIGURE OUT WHY IT ISNT TRAINING
                    self.q_hat_prime.load_state_dict(target_net_state_dict)
                
                if test_observation is not None:
                    with torch.no_grad():
                        q_values = self.q_hat(test_observation)
                        test_q_cache[self.steps_done-1] = q_values[0]
            
            # log episode reward
            self.tb_writer.add_scalar("mean_episode_reward", episode_reward / self.hyperparameters["EPISODE_LENGTH"], episode)
            self.tb_writer.add_histogram("action_count", action_count.flatten(), episode)
            # self.tb_writer.flush()
        
        self.tb_writer.flush()
        self.tb_writer.close()

        if test_observation is not None:
            return test_q_cache.cpu().numpy()
    
    def benchmark(self, n_episodes=10):
        rewards_cache = []
        for episode in range(n_episodes):
            # reset the enviroment tensordict
            td = self.env.reset(self.env.gen_params(batch_size=[self.hyperparameters["STEP_BATCH_SIZE"]], device=device))

            episode_reward = 0

            for step in range(self.hyperparameters["EPISODE_LENGTH"]):
                
                # get the current state
                observation = td["observation"]

                # select action
                actions, n_repair = self.select_action(td, entropy=0.0)

                # set the action in the tensordict
                td["action"] = actions

                # step the enviroment
                td = self.env.step(td)

                # move tensordict to next state
                td = td["next"]

                # get rewards
                rewards = td["reward"].squeeze()

                episode_reward += rewards.mean().cpu().item()
            
            rewards_cache.append(episode_reward / self.hyperparameters["EPISODE_LENGTH"])
        
        return sum(rewards_cache) / n_episodes

    def benchmark_UC(self, n_episodes=10, episode_length = None):
            orm_cache = []
            util_cache = []
            gamma = self.hyperparameters["GAMMA"]

            if episode_length is None:
                episode_length = self.hyperparameters["EPISODE_LENGTH"]

            for episode in range(n_episodes):
                # reset the enviroment tensordict
                td = self.env.reset(self.env.gen_params(batch_size=[self.hyperparameters["STEP_BATCH_SIZE"]], device=device))

                episode_reward = 0

                episode_orm = 0
                episode_util = 0

                gamma_weight_cache = 0

                for step in range(episode_length):
                    
                    # get the current state
                    observation = td["observation"]

                    # select action
                    actions, n_repair = self.select_action(td, entropy=0.0)

                    # set the action in the tensordict
                    td["action"] = actions

                    # step the enviroment
                    td = self.env.step(td)

                    # move tensordict to next state
                    td = td["next"]

                    # get rewards
                    rewards = td["reward"].squeeze()

                    # episode_reward += rewards.mean().cpu().item()
                    episode_orm += td["orm_costs"].mean().cpu().item() * (gamma ** step)
                    episode_util += td["utility"].mean().cpu().item() * (gamma ** step)
                    gamma_weight_cache += gamma ** step
                
                # rewards_cache.append(episode_reward / self.hyperparameters["EPISODE_LENGTH"])
                orm_cache.append(episode_orm / gamma_weight_cache)
                util_cache.append(episode_util / gamma_weight_cache)

            return sum(orm_cache) / n_episodes, sum(util_cache) / n_episodes
    
    def rollout(self, n_envs=4, n_steps=None):
        if n_steps is None:
            n_steps = self.hyperparameters["EPISODE_LENGTH"]

        # reset the enviroment tensordict
        td = self.env.reset(self.env.gen_params(batch_size=[n_envs], device=device))

        memory = TensorDict({}, [n_steps])

        for step in range(n_steps):
            # select action
            actions, n_repair = self.select_action(td, entropy=0.0)

            # set the action in the tensordict
            td["action"] = actions

            td = self.env.step(td)

            memory[step] = td

            # step the enviroment
            td = step_mdp(td)
        
        return memory
        