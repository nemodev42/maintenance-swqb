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

transition_tensor, costs_tensor, utility_tensor, hyperparameters= None, None, None, None

# def load_dynamics( transition, rsa, hyper):
#     global transition_tensor, rsa_tensor, hyperparameters
#     transition_tensor = transition
#     rsa_tensor = rsa
#     hyperparameters = hyper

def load_dynamics( transition, costs, utility, hyper):
    global transition_tensor, costs_tensor, utility_tensor, hyperparameters
    transition_tensor = transition
    costs_tensor = costs
    utility_tensor = utility
    hyperparameters = hyper



def _step(tensordict,):
    # global max_cost
    # import state information from the tensordict
    conditions = tensordict["conditions"]
    actions = tensordict["action"]
    #  Get the transition matrix and reward function from enviroment parameters
    transitions = tensordict["params", "transitions"]
    # rsa = tensordict["params", "rewards"]

    step = tensordict["step"]
    
    # Determine batch size, defaulting to 1 if there's no batch
    batch_size = actions.shape[0] if actions.dim() > 1 else 1

    # Gather transition probabilities based on batch size
    if batch_size == 1:
        transition_probs = transitions[
            torch.arange(hyperparameters["N_COMPONENTS"]),
            actions, conditions
        ]
    else:
        transition_probs = transitions[
            torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, hyperparameters["N_COMPONENTS"]),
            torch.arange(hyperparameters["N_COMPONENTS"], device=device).unsqueeze(0).expand(batch_size, hyperparameters["N_COMPONENTS"]),
            actions, conditions
        ]

    # Reshape transition_probs to [batch_size * N_COMPONENTS, 6]
    transition_probs_flat = transition_probs.view(-1, transition_probs.shape[-1])

    # Sample new conditions
    new_conditions = torch.multinomial(transition_probs_flat, 1).squeeze(-1)

    # Handle batch_size == 1 case separately to prevent shape mismatch
    if batch_size == 1:
        new_conditions = new_conditions.view(hyperparameters["N_COMPONENTS"])  # No batch dimension when batch_size = 1
    else:
        # Reshape back to the original batch structure for batch_size > 1
        new_conditions = new_conditions.view(batch_size, hyperparameters["N_COMPONENTS"])

    # Ensure no extra singleton dimension remains
    new_conditions = new_conditions.squeeze()  # Remove any singleton dimensions

    # Calculate rewards
    costs = tensordict["params", "repair_costs"]
    utility = tensordict["params", "utility"]
    blend = hyperparameters["REWARD_BLEND"]

    # Indexing for batch dimension handling
    # if batch_size == 1:
    #     rewards = rewards = rsa[
    #         torch.arange(hyperparameters["N_COMPONENTS"], device=device).unsqueeze(0).expand(batch_size, hyperparameters["N_COMPONENTS"]),
    #         conditions, actions
    #     ].sum(dim=-1)
    # else:
    #     rewards = rsa[
    #         torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, hyperparameters["N_COMPONENTS"]),
    #         torch.arange(hyperparameters["N_COMPONENTS"], device=device).unsqueeze(0).expand(batch_size, hyperparameters["N_COMPONENTS"]),
    #         conditions, actions
    #     ].sum(dim=-1)

    orm_costs, util_val = None, None

    if batch_size == 1:
        n_repair = torch.sum(actions)
        n_failed = torch.sum(new_conditions == hyperparameters["N_CONDITION_STATES"] - 1)
        orm_costs = costs[n_repair]
        util_val = utility[-n_failed]
        rewards = blend * util_val- (1-blend) * orm_costs
    else:
        n_repair = torch.sum(actions, dim=1)
        n_failed = torch.sum(new_conditions == hyperparameters["N_CONDITION_STATES"] - 1, dim=1)

        # print(f"n_repair: {n_repair.shape}")
        # print(f"n_failed: {n_failed.shape}")

        orm_costs = costs[torch.arange(batch_size, device=device), n_repair]
        util_val = utility[torch.arange(batch_size, device=device), hyperparameters["N_COMPONENTS"]-n_failed-1]

        # print(f"orm_costs: {orm_costs.shape}")
        # print(f"util_costs: {util.shape}")

        rewards = blend * util_val- (1-blend) * orm_costs

    # print(f"rewards: {rewards}")

    # Normalize rewards by max cost
    # rewards = rewards / max_cost

    # Set done to 0.
    done = torch.zeros_like(rewards, dtype=torch.bool)

    # Return new state tensordict
    out = TensorDict(
        {
            "conditions":new_conditions,
            "orm_costs":orm_costs,
            "utility":util_val,
            "params": tensordict["params"],
            "reward": rewards,
            "done": done,
            "step": step + 1,
        },
        tensordict.shape
    )
    return out



# Returns reset tensordict
def _reset(self, tensordict, randomize: bool = False):
    # If no tensordict is provided, generate one
    # assert tensordict is not None, "params tensordict must be provided for this environment"

    if tensordict is None or tensordict.is_empty():
        # Generate enviroment parameters
        tensordict = self.gen_params(batch_size=self.batch_size, device=self.device)


    # for non batch-locked environments, the input ``tensordict`` shape dictates the number
    # of simulators run simultaneously. In other contexts, the initial
    # random state's shape will depend upon the environment batch-size instead.
    
    # Get the shape of the tensordict
    conditions_shape = tensordict.shape 

    # add dimension for number of components
    conditions_shape = conditions_shape + (self.hyperparameters["N_COMPONENTS"],)

    # Initialize conditions to zeros
    conditions = ( torch.zeros(conditions_shape, dtype=torch.int64, device=self.device)  )

    # Randomize initial state if requested
    if randomize:
        conditions = torch.randint(0, self.hyperparameters["N_CONDITION_STATES"], conditions_shape, device=self.device)
        

    orm_costs = torch.zeros(tensordict.shape, dtype=torch.float32, device=self.device)
    utility = torch.zeros(tensordict.shape, dtype=torch.float32, device=self.device)
    step = torch.zeros(tensordict.shape, dtype=torch.float32, device=self.device)

    # Return the initial state
    out = TensorDict(
        {
            "conditions":conditions,
            "orm_costs":orm_costs,
            "utility":utility,
            "step":step,
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    return out



def _make_spec(self, td_params):
    # Under the hood, this will populate self.output_spec["observation"]
    self.observation_spec = Composite(
        conditions=Bounded(
            low=0,
            high=self.hyperparameters["N_CONDITION_STATES"],
            shape=(self.hyperparameters["N_COMPONENTS"],),
            dtype=torch.int64,
        ),
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
        orm_costs=Unbounded(shape=(), dtype=torch.float32), # add the costs to the observation spec
        utility=Unbounded(shape=(), dtype=torch.float32), # add the utility to the observation spec
        step=Unbounded(shape=(), dtype=torch.float32), # add the step to the observation spec
    )
    # since the environment is stateless, we expect the previous output as input.
    # For this, ``EnvBase`` expects some state_spec to be available
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    self.action_spec = Bounded(
        low=0,
        high=2,
        shape=(self.hyperparameters["N_COMPONENTS"],),
        dtype=torch.int64,
    )
    self.reward_spec = Unbounded(shape=(*td_params.shape, 1))

# for params spec
def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else Unbounded(
                dtype=tensor.dtype,  shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite



def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    # put on device
    rng = torch.Generator(device=self.device).manual_seed(seed)
    self.rng = rng



# Generates static enviroment parameters
def gen_params(  batch_size=None, device=device) -> TensorDictBase:
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    # "hyperparameters": hyperparameters,
                    "transitions": transition_tensor, # controls component degredation
                    # "rewards": rsa_tensor, # controls reward
                    "repair_costs": costs_tensor, # controls repair costs
                    "utility": utility_tensor, # controls utility
                },
                [],
            )
        },
        [],
    ).to(device=device)
    if batch_size:
        td = td.expand(batch_size).contiguous() # expand for batch
    return td



# Envicomrnt class declaration
class DiscreteMaitenanceEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self,  td_params=None, seed=None, device=device):
        self.hyperparameters = hyperparameters
        
        
        if td_params is None:
            td_params = self.gen_params( device=device)

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed







# calculates state distrobution
# class GeneralizationTransform(Transform):
#     def _apply_transform(self, obs: torch.Tensor) -> None:
#         batch_size = obs.shape[0]
#         # obs shape is [batch_size, N_Components] and is an integer 1-6
#         # Convert the obs to [batch_size, N_condtion_states] 
#         # and is the fraction of components in each condition for each paralell batched enviroment
#         obs = torch.nn.functional.one_hot(obs, num_classes=hyperparameters["N_CONDITION_STATES"]).float()
#         obs = obs.sum(dim=1)
#         obs = obs / obs.sum(dim=1, keepdim=True)
#         return obs

#     # The transform must also modify the data at reset time
#     def _reset(
#         self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
#     ) -> TensorDictBase:
#         return self._call(tensordict_reset)

#     # _apply_to_composite will execute the observation spec transform across all
#     # in_keys/out_keys pairs and write the result in the observation_spec which
#     # is of type ``Composite``
#     @_apply_to_composite
#     def transform_observation_spec(self, observation_spec):
#         return BoundedTensorSpec(
#             low=0,
#             high=1,
#             shape=(6,),
#             dtype=torch.float32,
#             device=device, #observation_spec.device,
#         )

# def transform_maitenance_env(env):
#     # Add the transform to the environment
#     t_generalize = GeneralizationTransform(in_keys=["conditions"], out_keys=["condition_distrobution"])
#     env = env.append_transform(t_generalize)
#     # turn the condition distrobution into a single tensor for the observation
#     cat_transform = CatTensors(
#         in_keys=["condition_distrobution"], dim=-1, out_key="observation", del_keys=False
#     )
#     env = env.append_transform(cat_transform)
#     return env


# # calculates state distrobution
# class GeneralizationTransform(Transform):
#     def _apply_transform(self, obs: torch.Tensor) -> None:
#         batch_size = obs.shape[0]
#         # print(obs.shape)
#         # obs shape is [batch_size, N_Components] and is an integer 1-6
#         # Convert the obs to [batch_size, N_condtion_states] 
#         # and is the fraction of components in each condition for each paralell batched enviroment
#         obs = torch.nn.functional.one_hot(obs, num_classes=hyperparameters["N_CONDITION_STATES"]).float()
#         obs = obs.sum(dim=1)
#         obs = obs / obs.sum(dim=1, keepdim=True)
#         # add the step to the obs
#         # obs = torch.cat([obs, step.unsqueeze(1)/hyperparameters["EPISODE_LENGTH"]], dim=1)
#         return obs

#     # The transform must also modify the data at reset time
#     def _reset(
#         self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
#     ) -> TensorDictBase:
#         return self._call(tensordict_reset)

#     # _apply_to_composite will execute the observation spec transform across all
#     # in_keys/out_keys pairs and write the result in the observation_spec which
#     # is of type ``Composite``
#     @_apply_to_composite
#     def transform_observation_spec(self, observation_spec):
#         return BoundedTensorSpec(
#             low=0,
#             high=1,
#             shape=(6,),
#             dtype=torch.float32,
#             device=device, #observation_spec.device,
#         )
    

# def transform_maitenance_env(env):
#     # Add the transform to the environment
#     t_generalize = GeneralizationTransform(in_keys=["conditions"], out_keys=["condition_distrobution"])
#     env = env.append_transform(t_generalize)
#     # turn the condition distrobution into a single tensor for the observation
#     cat_transform = CatTensors(
#         in_keys=["condition_distrobution"], dim=-1, out_key="observation", del_keys=False
#     )
#     env = env.append_transform(cat_transform)
#     return env



# from torchrl.envs.transforms import Transform
# from torchrl.data import TensorDictBase
# import torch
# import torch.nn.functional as F
# from torchrl.envs.utils import _apply_to_composite
# from torchrl.data.tensor_specs import BoundedTensorSpec

class GeneralizationTransform(Transform):
    def __init__(self, in_keys, out_keys, episode_length):
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.episode_length = episode_length

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        conditions = tensordict.get("conditions")  # [batch_size, N_Components]
        step = tensordict.get("step")              # [batch_size] or [batch_size, 1]
        
        # Create condition distribution
        batch_size = conditions.shape[0]
        condition_onehot = F.one_hot(conditions, num_classes=hyperparameters["N_CONDITION_STATES"]).float()
        condition_distrobution = condition_onehot.sum(dim=1)
        condition_distrobution = condition_distrobution / condition_distrobution.sum(dim=1, keepdim=True)

        # Normalize step
        step_fraction = step.unsqueeze(-1) / self.episode_length  # Make sure step is [batch_size, 1]

        # Concatenate
        full_observation = torch.cat([condition_distrobution, step_fraction], dim=-1)

        # Save to out_keys
        tensordict.set(self.out_keys[0], full_observation)
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            low=0,
            high=1,
            shape=(hyperparameters["N_CONDITION_STATES"] + 1,),
            dtype=torch.float32,
            device=observation_spec.device,
        )


def transform_maitenance_env(env):
    # Add the transform to the environment
    t_generalize = GeneralizationTransform(
        in_keys=["conditions"],
        out_keys=["observation"],
        episode_length=hyperparameters["EPISODE_LENGTH"]
    )
    env = env.append_transform(t_generalize)
    return env



def generate_maitenance_env(seed=None, device=device):

    assert hyperparameters is not None, "Please load the hyperarameters before creating the enviroment"

    # Create the environment
    env = DiscreteMaitenanceEnv(device=device,  seed=seed)

    env = transform_maitenance_env(env)

    return env

def reset_maitenance_env(env,):
    return env.reset( env.gen_params(  batch_size=[ hyperparameters["STEP_BATCH_SIZE"] ], device=device ) )