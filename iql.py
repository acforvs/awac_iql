from __future__ import annotations

import os
import fire
import yaml
import copy
import random
from dacite import from_dict
from dataclasses import dataclass, asdict
from typing import Callable
from functools import partial

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

import gym
import d4rl

import numpy as np


@dataclass
class PreprocessingConfig:
    normalize_reward: bool = True
    std_eps: float = 1e-3


@dataclass
class TrainingConfig:
    env_name: str = "antmaze-medium-play-v0"
    save_path: str = ""
    batch_size: int = 256
    buffer_size: int = 2_000_000
    max_steps: int = 1_000_000
    seed: int = 0


@dataclass
class EvalConfig:
    evaluate_every_n: int = 5_000
    eval_episodes: int = 100


@dataclass
class FineTuneConfig:
    batch_size: int = 256
    max_steps: int = 1_000_000
    save_path: str = ""


@dataclass
class IQLConfig:
    alpha: float = 0.005
    beta: float = 10.0
    tau: float = 0.9
    gamma: float = 0.99
    max_weight: float = 100.0


@dataclass
class TrainConfig:
    project: str = "TLab-Application-v0"
    group: str = "IQL"
    name: str = "IQL"
    preprocess: PreprocessingConfig = PreprocessingConfig()
    eval: EvalConfig = EvalConfig()
    train: TrainingConfig = TrainingConfig()
    finetune: FineTuneConfig = FineTuneConfig()
    algorithm: IQLConfig = IQLConfig()


def seed_everything(
    seed: int, env: gym.Env | None = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def asymetric_loss(sample: torch.Tensor, tau: float) -> torch.Tensor:
    loss = (tau - (sample < 0).float()).abs() * sample**2
    return loss.mean()


TensorBatch = list[torch.Tensor]


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        device: str = "cpu",
    ):
        self._capacity = capacity
        self._device = device
        self._size = 0
        self._ptr = 0

        self._states = self._zeros_tensor(shape=(capacity, state_dim))
        self._actions = self._zeros_tensor(shape=(capacity, action_dim))
        self._rewards = self._zeros_tensor(shape=(capacity, 1))
        self._next_states = self._zeros_tensor(shape=(capacity, state_dim))
        self._dones = self._zeros_tensor(shape=(capacity, 1))

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def _zeros_tensor(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: dict[str, np.ndarray], info: bool = True) -> None:
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._capacity:
            raise ValueError(
                f"Buffer capacity is smaller than the size of the dataset: {self._capacity} < {n_transitions}"
            )

        if info:
            print(f"Loading the dataset of size {n_transitions}...")

        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size, self._ptr = n_transitions, n_transitions

        if info:
            print(f"Successfuly loaded. Size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        inds = np.random.randint(0, self._size, size=batch_size)
        states, actions, rewards, next_states, dones = (
            self._states[inds],
            self._actions[inds],
            self._rewards[inds],
            self._next_states[inds],
            self._dones[inds],
        )
        return [states, actions, rewards, next_states, dones]

    def insert(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: float,
        next_state: np.ndarray,
    ) -> None:
        self._states[self._ptr] = self._to_tensor(state)
        self._actions[self._ptr] = self._to_tensor(action)
        self._rewards[self._ptr] = float(reward)
        self._dones[self._ptr] = float(done)
        self._next_states[self._ptr] = self._to_tensor(next_state)

        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")
        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_fn is not None:
            layers.append(output_fn())
        if squeeze_output and dims[-1] != 1:
            raise ValueError("Last dim must be 1 when squeezing")
        if squeeze_output:
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasePolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_hidden: int,
        max_action: float,
    ):
        super().__init__()
        self.max_action = max_action
        self.net = MLP(
            dims=(state_dim, *[hidden_dim for _ in range(n_hidden)], action_dim),
            output_fn=nn.Tanh,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def _extract_action(self, action: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        action = self(state)
        action = self._extract_action(action)
        return (
            torch.clamp(action * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class GaussianPolicy(BasePolicy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ):
        super().__init__(state_dim, action_dim, hidden_dim, n_hidden, max_action)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def _extract_action(self, action: torch.Tensor) -> torch.Tensor:
        return action.mean if not self.training else action.sample()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        mu = self.net(state)
        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))
        return MultivariateNormal(mu, scale_tril=torch.diag(std))


class DeterministicPolicy(BasePolicy):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.nex(state)

    def _extract_action(self, action: torch.Tensor) -> torch.Tensor:
        return action


class Critic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        self.q1 = MLP(
            dims=(state_dim + action_dim, *[hidden_dim for _ in range(n_hidden)], 1),
            squeeze_output=True,
        )
        self.q2 = copy.deepcopy(self.q1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor]:
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


class ValueCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        self.value = MLP(
            dims=(state_dim, *[hidden_dim for _ in range(n_hidden)], 1),
            squeeze_output=True,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value(state)


class IQL(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def setup(
        self,
        actor: BasePolicy,
        actor_optim_cls: torch.optim.Optimizer,
        actor_optim_kwargs: dict,
        actor_scheduler: torch.optim.lr_scheduler.LRScheduler,
        critic: Critic,
        critic_optim_cls: torch.optim.Optimizer,
        critic_optim_kwargs: dict,
        value_critic: ValueCritic,
        value_optim_cls: torch.optim.Optimizer,
        value_optim_kwargs: dict,
        alpha: float = 0.005,
        beta: float = 3.0,
        gamma: float = 0.99,
        tau: float = 0.7,
        max_weight=100.0,
        total_max_steps: int = 1_000_000,
        device: str = "cpu",
        **kwargs,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.asymetric_loss = partial(asymetric_loss, tau=tau)
        self.device = device
        self.max_steps = total_max_steps
        self.steps = 0

        self._max_weight = max_weight

        self.critic = critic.to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.critic_optim = critic_optim_cls(
            self.critic.parameters(), **critic_optim_kwargs
        )

        self.value_critic = value_critic.to(device)
        self.value_optim = value_optim_cls(
            self.value_critic.parameters(), **value_optim_kwargs
        )

        self.actor = actor.to(device)
        self.actor_optim = actor_optim_cls(
            self.actor.parameters(), **actor_optim_kwargs
        )
        self.actor_scheduler = actor_scheduler(self.actor_optim, total_max_steps)

        self._setup_is_called = True

    def _update_v(self, states: torch.Tensor, actions: torch.Tensor, logger_info: dict):
        with torch.no_grad():
            target_q = torch.min(*self.target_critic(states, actions))

        value = self.value_critic(states)
        advantage = target_q - value
        loss = self.asymetric_loss(advantage)
        logger_info["value_critic_loss"] = loss.item()
        self.value_optim.zero_grad()
        loss.backward()
        self.value_optim.step()
        return advantage

    def _update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_v: torch.Tensor,
        logger_info: dict,
    ):
        objective = rewards + (1.0 - dones.float()) * next_v.detach() * self.gamma
        qs = self.critic(states, actions)
        loss = sum(F.mse_loss(q, objective) for q in qs) / len(qs)
        logger_info["critic_loss"] = loss.item()
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def _soft_update(self, target_net, source_net):
        for target_parameter, source_parameter in zip(
            target_net.parameters(), source_net.parameters()
        ):
            target_parameter.data.mul_(1 - self.alpha)
            target_parameter.data.add_(self.alpha * source_parameter.data)

    def _extract_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantage: torch.Tensor,
        logger_info: dict,
    ):
        # And following Brandfonbrener et al. (2021) we clip exponentiated advantages to (−∞, 100].
        exp = torch.exp(self.beta * advantage.detach()).clamp(max=self._max_weight)
        policy = self.actor(states)
        if isinstance(policy, torch.distributions.Distribution):
            bc_loss = -policy.log_prob(actions)
        elif torch.is_tensor(policy):
            bc_loss = torch.sum((policy - actions) ** 2, dim=1)
        loss = torch.mean(exp * bc_loss)
        logger_info["actor_loss"] = loss.item()
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        self.actor_scheduler.step()

    def update(self, batch: TensorBatch) -> dict:
        if not self._setup_is_called:
            raise RuntimeError("Setup is not called, cannot proceed with training")

        self.steps += 1
        logger_info = {}

        states, actions, rewards, next_states, dones = batch
        with torch.no_grad():
            next_v = self.value_critic(next_states)
        advantage = self._update_v(states, actions, logger_info)
        self._update_q(
            states,
            actions,
            rewards.squeeze(dim=-1),
            dones.squeeze(dim=-1),
            next_v,
            logger_info,
        )
        self._extract_policy(states, actions, advantage, logger_info)
        self._soft_update(self.target_critic, self.critic)

        return logger_info

    @torch.no_grad()
    def act(self, state: np.ndarray):
        return self.actor.act(state, self.device)

    def _state_objects(self):
        return {
            "actor": self.actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "value_critic": self.value_critic,
            "actor_optim": self.actor_optim,
            "critic_optim": self.critic_optim,
            "value_optim": self.value_optim,
            "actor_scheduler": self.actor_scheduler,
        }

    def state_dict(self) -> dict:
        state = {"steps": self.steps}
        objects = self._state_objects()
        for key in objects:
            state[key] = objects[key].state_dict()
        return state

    def load_state_dict(self, state: dict) -> None:
        self.steps = state["steps"]
        objects = self._state_objects()
        for key in objects:
            objects[key].load_state_dict(state[key])


class BaseLogger:
    def log(self, data: dict, step: int):
        raise NotImplemented


class DummyWandbLogger(BaseLogger):
    def __init__(self, config: TrainConfig):
        wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            config=asdict(config),
        )

    def log(self, data: dict, step: int):
        wandb.log(data, step=step)


class PrintLogger(BaseLogger):
    def log(self, data, step: int):
        print(f"Step {step}: {data}")


class EvaluationMixin:
    """
    Adds functionality of a policy evaluation
    The following methods must be defined:
        self._env: gym.Env, current environment
        self._logger: None | BaseLogger, adds logging functionality (reports normalized_score)
        self.steps: int, current number of steps, required if self._logger is not None
        self.eval: func, the base class must be inherited from nn.Module
        self.train: func, the base class must be inherited from nn.Module
        self.act, func, selects the next action based on an observation
    """

    @torch.no_grad()
    def evaluate(self, n_episodes: int, seed: int) -> None:
        self.eval()
        episode_rewards = []
        for _ in range(n_episodes):
            state, done = self._env.reset(), False
            episode_reward = 0.0
            while not done:
                state, reward, done, _ = self._env.step(self.act(state))
                episode_reward += reward
            episode_rewards.append(episode_reward)
        self.train()
        score = np.asarray(episode_rewards).mean()
        normalized_eval_score = self._env.get_normalized_score(score) * 100.0
        if self._logger is not None:
            self._logger.log(
                {"normalized_score": normalized_eval_score}, step=self.steps
            )

    @torch.no_grad()
    def act(self, state: np.ndarray):
        raise NotImplemented


class OfflinePretrainMixin(EvaluationMixin):
    """
    Adds functionality of an offline pretrain
    Adds the following attributes:
        self._env: gym.Env, initialized during .setup_env
        self._action_dim & self._state_dim: int, initialized during .setup_env, dims
        self._logger: BaseLogger | None, initialized during .run_offline
        self._replay_buffer: ReplayBuffer, initialized during .run_offline
    The following methods must be defined:
        self.act: func, selects the next action based on an observation
        self.update: func, updates the algorithm, takes single batch from replay buffer
        self.steps: int, current number of steps, required if self._logger is not None
    """

    @staticmethod
    def _reward_range(dataset, max_episode_steps: int):
        returns, lengths = [], []
        ep_ret, ep_len = 0.0, 0
        for reward, done in zip(dataset["rewards"], dataset["terminals"]):
            ep_ret += float(reward)
            ep_len += 1
            if done or ep_len == max_episode_steps:
                returns.append(ep_ret)
                lengths.append(ep_len)
                ep_ret, ep_len = 0.0, 0
        lengths.append(ep_len)
        return min(returns), max(returns)

    def _modify_reward(self, dataset, env_name: str, max_episode_steps=1000):
        if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
            min_ret, max_ret = self._reward_range(dataset, max_episode_steps)
            dataset["rewards"] /= max_ret - min_ret
            dataset["rewards"] *= max_episode_steps
        elif "antmaze" in env_name:
            dataset["rewards"] -= 1.0

    def _preprocess_dataset(
        self,
        dataset,
        env_name: str,
        normalize_reward: bool,
        std_eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if normalize_reward:
            self._modify_reward(dataset, env_name)
        state_mean, state_std = (
            dataset["observations"].mean(0),
            dataset["observations"].std(0) + std_eps,
        )
        for key in ("observations", "next_observations"):
            dataset[key] = (dataset[key] - state_mean) / state_std

        return state_mean, state_std

    def _wrap_env(
        self,
        env: gym.Env,
        state_mean: np.ndarray | float,
        state_std: np.ndarray | float,
    ) -> gym.Env:
        def normalize_state(state):
            return (state - state_mean) / state_std

        return gym.wrappers.TransformObservation(env, normalize_state)

    def setup_env(self, config: TrainConfig):
        self._env = gym.make(config.train.env_name)
        self._state_dim = self._env.observation_space.shape[0]
        self._action_dim = self._env.action_space.shape[0]
        max_action = float(self._env.action_space.high[0])

        return self._state_dim, self._action_dim, max_action

    def run_offline(
        self,
        config: TrainConfig,
        logger: BaseLogger | None = None,
    ):
        self._logger = logger

        preprocessing_config, evaluation_config, training_config = (
            config.preprocess,
            config.eval,
            config.train,
        )

        seed = training_config.seed
        seed_everything(seed)

        dataset = d4rl.qlearning_dataset(self._env)
        mu, std = self._preprocess_dataset(
            dataset,
            training_config.env_name,
            **asdict(preprocessing_config),
        )
        self._env = self._wrap_env(self._env, mu, std)

        self._replay_buffer = ReplayBuffer(
            self._state_dim,
            self._action_dim,
            training_config.buffer_size,
            self.device,
        )
        self._replay_buffer.load_d4rl_dataset(dataset)

        evaluate_every_n = evaluation_config.evaluate_every_n
        eval_episodes = evaluation_config.eval_episodes

        for step_n in range(training_config.max_steps):
            batch = self._replay_buffer.sample(training_config.batch_size)
            logger_info = self.update(batch)
            if self._logger is not None:
                self._logger.log(logger_info, step=self.steps)

            if evaluate_every_n is not None and (step_n + 1) % evaluate_every_n == 0:
                self.evaluate(eval_episodes, seed)

    def update(self, batch: TensorBatch) -> dict[str]:
        raise NotImplemented

    @torch.no_grad()
    def act(self, state: np.ndarray):
        raise NotImplemented


class OnlineFineTuneMixin(EvaluationMixin):
    """
    Adds functionality of an online finetuning
    The following attributed must be defined:
        self._env: gym.Env
        self._logger: BaseLogger | None
        self._replay_buffer: ReplayBuffer
    The following methods must be defined:
        self.act: func, selects the next action based on an observation
        self.update: func, updates the algorithm, takes single batch from replay buffer
        self.steps: int, current number of steps, required if self._logger is not None
    """

    def run_online(self, config: TrainConfig, logger: BaseLogger | None = None):
        self._logger = logger
        evaluation_config, finetuning_config = config.eval, config.finetune

        seed = config.train.seed
        evaluate_every_n = evaluation_config.evaluate_every_n
        eval_episodes = evaluation_config.eval_episodes

        state, done = self._env.reset(), False
        for step_n in range(finetuning_config.max_steps):
            action = self.act(state)
            next_state, reward, done, _ = self._env.step(action)
            self._replay_buffer.insert(state, action, reward, float(done), next_state)
            batch = self._replay_buffer.sample(finetuning_config.batch_size)
            logger_info = self.update(batch)
            if self._logger is not None:
                self._logger.log(logger_info, step=self.steps)
            state = next_state
            if done:
                state, done = self._env.reset(), False

            if evaluate_every_n is not None and (step_n + 1) % evaluate_every_n == 0:
                self.evaluate(eval_episodes, seed)

    @torch.no_grad()
    def act(self, state: np.ndarray):
        raise NotImplemented


class IQLOfflineOnline(IQL, OnlineFineTuneMixin, OfflinePretrainMixin):
    ...


def train(yaml_path: str = ""):
    if not yaml_path:
        config = TrainConfig()
    else:
        with open(yaml_path) as f:
            options = yaml.load(f, Loader=yaml.SafeLoader)
            config = from_dict(TrainConfig, options)

    iql_trainer = IQLOfflineOnline()
    state_dim, action_dim, max_action = iql_trainer.setup_env(config)
    iql_trainer.setup(
        actor=GaussianPolicy(state_dim, action_dim, max_action=max_action),
        actor_optim_cls=torch.optim.Adam,
        actor_optim_kwargs={"lr": 3e-4},
        critic=Critic(state_dim, action_dim),
        critic_optim_cls=torch.optim.Adam,
        critic_optim_kwargs={"lr": 3e-4},
        value_critic=ValueCritic(state_dim),
        value_optim_cls=torch.optim.Adam,
        value_optim_kwargs={"lr": 3e-4},
        actor_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        total_max_steps=config.train.max_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **asdict(config.algorithm),
    )
    wandb_logger = DummyWandbLogger(config)
    iql_trainer.run_offline(config, wandb_logger)
    if config.train.save_path:
        torch.save(iql_trainer.state_dict(), config.train.save_path)
    iql_trainer.run_online(config, wandb_logger)
    if config.finetune.save_path:
        torch.save(iql_trainer.state_dict(), config.finetune.save_path)


if __name__ == "__main__":
    fire.Fire(train)
