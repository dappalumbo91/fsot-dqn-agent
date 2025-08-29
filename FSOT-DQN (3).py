import mpmath as mp
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gc # For garbage collection
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
mp.mp.dps = 15 # Reduced precision for speed
# FSOT Modulator Module (neural plasticity/synaptic adjustment, with causal intervention, per-dim modulation, and oscillations)
class FSOT_Modulator:
    def __init__(self):
        self.phi = (1 + mp.sqrt(5)) / 2
        self.e = mp.e
        self.pi = mp.pi
        self.sqrt2 = mp.sqrt(2)
        self.log2 = mp.log(2)
        self.gamma_euler = mp.euler
        self.catalan_G = mp.catalan
        self.alpha = mp.log(self.pi) / (self.e * self.phi**13)
        self.psi_con = (self.e - 1) / self.e
        self.eta_eff = 1 / (self.pi - 1)
        self.beta = 1 / mp.exp(self.pi**self.pi + (self.e - 1))
        self.gamma = -self.log2 / self.phi
        self.omega = mp.sin(self.pi / self.e) * self.sqrt2
        self.theta_s = mp.sin(self.psi_con * self.eta_eff)
        self.poof_factor = mp.exp(-(mp.log(self.pi) / self.e) / (self.eta_eff * mp.log(self.phi)))
        self.acoustic_bleed = mp.sin(self.pi / self.e) * self.phi / self.sqrt2
        self.phase_variance = -mp.cos(self.theta_s + self.pi)
        self.coherence_efficiency = (1 - self.poof_factor * mp.sin(self.theta_s)) * (1 + 0.01 * self.catalan_G / (self.pi * self.phi))
        self.bleed_in_factor = self.coherence_efficiency * (1 - mp.sin(self.theta_s) / self.phi)
        self.acoustic_inflow = self.acoustic_bleed * (1 + mp.cos(self.theta_s) / self.phi)
        self.suction_factor = self.poof_factor * -mp.cos(self.theta_s - self.pi)
        self.chaos_factor = self.gamma / self.omega
        self.perceived_param_base = self.gamma_euler / self.e
        self.new_perceived_param = self.perceived_param_base * self.sqrt2
        self.consciousness_factor = self.coherence_efficiency * self.new_perceived_param
    def compute_adjustment(self, N=10, P=0.9, D_eff=25, recent_hits=1, delta_psi=0.5, rho=0.8, observed=True, causal_intervention=0.0, dim=0, total_dims=3, steps=0):
        oscillation = mp.sin(steps * self.pi / 20) * 0.05 # Further reduced amp for less aggressive modulation
        delta_psi += causal_intervention + oscillation
        D_eff = D_eff * (1 + dim / total_dims) # Per-dim scaling
        dim_term = mp.sin(dim * self.pi / total_dims) * self.chaos_factor
        growth_term = mp.exp(self.alpha * (1 - recent_hits / N) * self.gamma_euler / self.phi)
        term1 = (N * P / mp.sqrt(D_eff)) * mp.cos((self.psi_con + delta_psi) / self.eta_eff) * mp.exp(-self.alpha * recent_hits / N + rho + self.bleed_in_factor * delta_psi) * (1 + growth_term * self.coherence_efficiency)
        perceived_adjust = 1 + self.new_perceived_param * mp.log(D_eff / 25)
        term1 *= perceived_adjust
        quirk_mod = 1
        if observed:
            quirk_mod = mp.exp(self.consciousness_factor * self.phase_variance) * mp.cos(delta_psi + self.phase_variance)
        term1 *= quirk_mod
        term2 = 1
        term3 = self.beta * mp.cos(delta_psi) * (N * P / mp.sqrt(D_eff)) * (1 + self.chaos_factor * (D_eff - 25) / 25) * (1 + self.poof_factor * mp.cos(self.theta_s + self.pi) + self.suction_factor * mp.sin(self.theta_s)) * (1 + self.acoustic_bleed * mp.sin(1)**2 / self.phi + self.acoustic_inflow * mp.cos(1)**2 / self.phi) * (1 + self.bleed_in_factor * self.phase_variance)
        adjustment = float(term1 + term2 + term3 + dim_term)
        return max(0.0, adjustment)  # Clip to non-negative to avoid worsening rewards
# Environment Module (sensory input/world interaction)
class ND_Env:
    def __init__(self, dims=3, size=5):
        self.dims = dims
        self.size = size
        self.state = tuple([0] * dims)
        self.goal = tuple([size-1] * dims)
        self.actions = [(i, d) for i in range(dims) for d in [-1, 1]]
    def step(self, action):
        dim, dir_val = action
        new_state = list(self.state)
        new_state[dim] = max(0, min(self.size-1, new_state[dim] + dir_val))
        new_state = tuple(new_state)
        reward = -0.2  # Further reduced base penalty
        old_dist = sum(self.size - 1 - s for s in self.state)
        new_dist = sum(self.size - 1 - s for s in new_state)
        if new_dist < old_dist:
            reward += 1.0  # Increased shaping reward
        done = new_state == self.goal
        if done: reward = 200  # Increased goal reward
        self.state = new_state
        return new_state, reward, done
# Memory Buffer Module (experience storage/memory recall, with offline prefill and prioritized consolidation)
class MemoryBuffer:
    def __init__(self, max_size=1000):
        self.memory = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size) # For prioritized replay
    def add(self, state, action, reward, next_state, done, priority=1.0):
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return []
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        return [self.memory[i] for i in indices]
    def prefill_offline(self, trajectories):
        for traj in trajectories:
            for exp in traj:
                self.add(*exp, priority=1.0)
    def consolidate(self, agent, top_k=0.1):
        if not self.memory:
            return
        sorted_indices = np.argsort(self.priorities)[::-1][:int(len(self.memory) * top_k)]
        for i in sorted_indices:
            state, action_idx, adjusted_reward, next_state, done = self.memory[i]
            dim = action_idx // 2 if len(self.memory) > 0 else 0 # Approximate dim from action_idx
            agent.train_step(state, action_idx, adjusted_reward, next_state, done, steps=0, causal_intervention=0, dim=dim, replay=True) # Replay with fixed params
# Policy Network Module (decision-making/Q-value approximation, with predictive coding and pruning)
class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, layers=[32, 32], sparsity_thresh=0.05):
        super(PolicyNet, self).__init__()
        self.layers = layers.copy()
        self.backbone = nn.ModuleList()
        self.backbone.append(nn.Linear(state_size, layers[0]))
        for i in range(1, len(layers)):
            self.backbone.append(nn.Linear(layers[i-1], layers[i]))
        self.q_head = nn.Linear(layers[-1], action_size)
        self.pred_head = nn.Linear(layers[-1], state_size)
        self.sparsity_thresh = sparsity_thresh
        self.masks = [None] * len(self.backbone)
    def forward(self, x):
        for i, layer in enumerate(self.backbone):
            x = torch.relu(layer(x))
        q_values = self.q_head(x)
        pred_state = self.pred_head(x)
        return q_values, pred_state
    def add_layer(self, new_neurons=32):
        last_out = self.layers[-1]
        self.layers.append(new_neurons)
        new_layer = nn.Linear(last_out, new_neurons)
        nn.init.kaiming_normal_(new_layer.weight, nonlinearity='relu')
        self.backbone.append(new_layer)
        self.masks.append(None)
        self.q_head = nn.Linear(new_neurons, self.q_head.out_features)
        nn.init.kaiming_normal_(self.q_head.weight, nonlinearity='linear')
        self.pred_head = nn.Linear(new_neurons, self.pred_head.out_features)
        nn.init.kaiming_normal_(self.pred_head.weight, nonlinearity='linear')
    def apply_pruning(self):
        for i, layer in enumerate(self.backbone):
            mask = (torch.abs(layer.weight.data) > self.sparsity_thresh).float()
            self.masks[i] = mask
            layer.weight.data *= mask
# Meta Controller for hierarchical RL (high-level sub-goal selection)
class MetaController:
    def __init__(self, state_size, sub_goal_size, layers=[16, 16], lr=0.001, gamma=0.99, epsilon=0.25, epsilon_decay=0.995, epsilon_min=0.01, batch_size=32, pred_weight=0.2):
        self.state_size = state_size
        self.action_size = sub_goal_size
        self.model = PolicyNet(state_size, sub_goal_size, layers)
        self.target_model = PolicyNet(state_size, sub_goal_size, layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=1000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.pred_weight = pred_weight
        self.update_target()
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def choose_sub_goal(self, state, current_dims, explore=True):
        pad_len = self.state_size - len(state)
        state_pad = list(state) + [0] * pad_len
        state_t = torch.FloatTensor([s / 4 for s in state_pad]).unsqueeze(0)
        if explore and np.random.random() <= self.epsilon:
            return np.random.randint(0, current_dims)
        with torch.no_grad():
            q_values, _ = self.model(state_t)
            q_values = q_values[0, :current_dims]
        return q_values.argmax().item()
    def store_transition(self, state, sub_goal, reward, next_state, done):
        pad_len = self.state_size - len(state)
        state_pad = tuple(list(state) + [0] * pad_len)
        next_state_pad = tuple(list(next_state) + [0] * pad_len)
        self.memory.append((state_pad, sub_goal, reward, next_state_pad, done))
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor([[s / 4 for s in st] for st in states])
        next_states = torch.FloatTensor([[s / 4 for s in st] for st in next_states])
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor([float(d) for d in dones])
        q_values, pred_states = self.model(states)
        q_values = q_values.gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_states)
            next_q_values = next_q_values.max(1)[0].detach()
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        q_loss = nn.MSELoss()(q_values, targets)
        pred_loss = nn.MSELoss()(pred_states, next_states)
        loss = q_loss + self.pred_weight * pred_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.model.apply_pruning()
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
# Agent Module (executive control/training integration, with safe constraints and predictive coding)
class FSOT_DQNAgent:
    def __init__(self, env, modulator, memory, meta_controller, lr=0.001, gamma=0.99, epsilon=0.15, epsilon_decay=0.999, epsilon_min=0.01, batch_size=16, layers=[32, 32], lambda_safe=0.0, pred_weight=0.1):
        self.env = env
        self.modulator = modulator
        self.memory = memory
        self.meta_controller = meta_controller
        self.state_size = env.dims
        self.action_size = len(env.actions)
        self.model = PolicyNet(self.state_size, self.action_size, layers)
        self.target_model = PolicyNet(self.state_size, self.action_size, layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.lambda_safe = lambda_safe
        self.pred_weight = pred_weight
        self.update_target()
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    def choose_action(self, state, sub_goal_dim):
        actions_in_dim = [i for i, a in enumerate(self.env.actions) if a[0] == sub_goal_dim]
        if np.random.random() <= self.epsilon:
            return random.choice(actions_in_dim)
        state = torch.FloatTensor([s / (self.env.size - 1) for s in state]).unsqueeze(0)
        with torch.no_grad():
            q_values, _ = self.model(state)
        q_in_dim = q_values[0, actions_in_dim]
        max_idx = q_in_dim.argmax().item()
        return actions_in_dim[max_idx]
    def train_step(self, state, action_idx, reward, next_state, done, steps, causal_intervention, dim, replay=False):
        if replay:
            states = torch.FloatTensor([[s / (self.env.size - 1) for s in state]])
            next_states = torch.FloatTensor([[s / (self.env.size - 1) for s in next_state]])
            actions = torch.LongTensor([action_idx]).unsqueeze(1)
            rewards = torch.FloatTensor([reward])
            dones = torch.FloatTensor([float(done)])
            q_values, pred_states = self.model(states)
            q_values = q_values.gather(1, actions).squeeze(1)
            with torch.no_grad():
                next_q_values, _ = self.target_model(next_states)
                next_q_values = next_q_values.max(1)[0].detach()
            targets = rewards + self.gamma * next_q_values * (1 - dones)
            q_loss = nn.MSELoss()(q_values, targets)
            pred_loss = nn.MSELoss()(pred_states, next_states)
            loss = q_loss + self.pred_weight * pred_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.apply_pruning()
        else:
            adjusted_reward = reward + max(0, self.modulator.compute_adjustment(causal_intervention=causal_intervention, dim=dim, total_dims=self.env.dims, steps=steps))  # Clip to positive
            # No safe_cost
            # Predictive coding surprise bonus
            with torch.no_grad():
                _, pred_state = self.model(torch.FloatTensor([s / (self.env.size - 1) for s in state]).unsqueeze(0))
                pred_error = nn.MSELoss()(pred_state, torch.FloatTensor([s / (self.env.size - 1) for s in next_state]).unsqueeze(0))
            surprise_bonus = pred_error.item() * 0.5  # Increased
            adjusted_reward += surprise_bonus
            # Calculate TD-error for priority
            with torch.no_grad():
                next_q_values, _ = self.target_model(torch.FloatTensor([s / (self.env.size - 1) for s in next_state]).unsqueeze(0))
                next_q = next_q_values.max(1)[0].item()
                current_q_values, _ = self.model(torch.FloatTensor([s / (self.env.size - 1) for s in state]).unsqueeze(0))
                current_q = current_q_values[0, action_idx].item()
            td_error = abs(adjusted_reward + self.gamma * next_q * (1 - float(done)) - current_q)
            self.memory.add(state, action_idx, adjusted_reward, next_state, done, priority=td_error + 1e-5)
            if steps % 8 == 0:
                minibatch = self.memory.sample(self.batch_size)
                if not minibatch:
                    return
                states, actions, rewards, next_states, dones = zip(*minibatch)
                states = torch.FloatTensor([[s / (self.env.size - 1) for s in st] for st in states])
                next_states = torch.FloatTensor([[s / (self.env.size - 1) for s in st] for st in next_states])
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor([float(d) for d in dones])
                q_values, pred_states = self.model(states)
                q_values = q_values.gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_q_values, _ = self.target_model(next_states)
                    next_q_values = next_q_values.max(1)[0].detach()
                targets = rewards + self.gamma * next_q_values * (1 - dones)
                q_loss = nn.MSELoss()(q_values, targets)
                pred_loss = nn.MSELoss()(pred_states, next_states)
                loss = q_loss + self.pred_weight * pred_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.apply_pruning()
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    def check_dynamic_growth(self, recent_rewards, optimal_reward):
        if len(recent_rewards) >= 5 and np.mean(recent_rewards) < 0.8 * optimal_reward:
            self.model.add_layer()
            self.target_model = PolicyNet(self.state_size, self.action_size, self.model.layers)
            self.optimizer = optim.Adam(self.model.parameters(), self.optimizer.param_groups[0]['lr'])
            self.update_target()
# Curriculum Module (learning progression)
class Curriculum:
    def __init__(self, start_dims=3, max_dims=3, size=5, threshold=0.3, window=5):
        self.current_dims = start_dims
        self.max_dims = max_dims
        self.size = size
        self.threshold = threshold
        self.window = window
    def get_env(self):
        return ND_Env(dims=self.current_dims, size=self.size)
    def check_progress(self, recent_rewards, optimal_reward):
        if len(recent_rewards) >= self.window and np.mean(recent_rewards) >= self.threshold * optimal_reward and self.current_dims < self.max_dims:
            self.current_dims += 1
            return True
        return False
# Generate Synthetic Data for Meta/Offline Pretraining (FSOT-based trajectories)
def generate_synthetic_trajectories(env, num_traj=300, modulator=None):
    trajectories = []
    for _ in range(num_traj):
        traj = []
        env.state = tuple([0] * env.dims) # Reset state for each traj
        state = env.state
        for _ in range(20):
            if np.random.random() < 0.7:
                low_dims = [i for i in range(env.dims) if state[i] < env.size - 1]
                if low_dims:
                    dim = np.random.choice(low_dims)
                    dir_val = 1
                    action = (dim, dir_val)
                    action_idx = env.actions.index(action)
                else:
                    action_idx = np.random.randint(0, len(env.actions))
            else:
                action_idx = np.random.randint(0, len(env.actions))
            action = env.actions[action_idx]
            next_state, reward, done = env.step(action)
            if modulator is not None:
                dim = action[0]
                steps = len(traj)
                causal_intervention = np.random.uniform(-0.1, 0.1)
                adjustment = modulator.compute_adjustment(steps=steps, causal_intervention=causal_intervention, dim=dim, total_dims=env.dims)
                adjusted = reward + max(0, adjustment)  # Clip in pretraining too
            else:
                adjusted = reward
            traj.append((state, action_idx, adjusted, next_state, done))
            state = next_state
            if done: break
        trajectories.append(traj)
    return trajectories
# Main Training (orchestration with all integrations)
curriculum = Curriculum()
modulator = FSOT_Modulator()
memory = MemoryBuffer()
rewards = []
recent_rewards = deque(maxlen=5)
# Offline/Meta Pretraining: Fill buffer with synthetic data
synthetic_traj = generate_synthetic_trajectories(curriculum.get_env(), modulator=modulator)
memory.prefill_offline(synthetic_traj)
# Count successful synthetic trajectories for logging
success_count = sum(any(exp[4] for exp in traj) for traj in synthetic_traj)
print(f"Pretraining: {success_count} successful trajectories out of {len(synthetic_traj)}")
max_dims = curriculum.max_dims
meta_controller = MetaController(state_size=max_dims, sub_goal_size=max_dims)
agent = None
prev_dims = None
consecutive_same_dim = 0
prev_sub_goal = -1
for episode in range(40):
    print(f"Episode {episode} started, Dims: {curriculum.current_dims}")
    env = curriculum.get_env()
    if agent is None or curriculum.current_dims != prev_dims:
        agent = FSOT_DQNAgent(env, modulator, memory, meta_controller, batch_size=16)
        prev_dims = curriculum.current_dims
    else:
        agent.env = env  # Update env reference if dims unchanged
    optimal_reward = 100 - (env.dims * 4 - 1) # Approx min steps reward
    state = tuple([0] * env.dims)
    env.state = state
    total_reward = 0
    done = False
    steps = 0
    causal_intervention = np.random.uniform(-0.1, 0.1) # Causal RL: Random intervention for testing cause-effect
    consecutive_same_dim = 0
    prev_sub_goal = -1
    while not done and steps < 200:
        sub_goal_dim = meta_controller.choose_sub_goal(state, env.dims)
        print(f"  Sub-goal dim: {sub_goal_dim}")
        sub_reward = 0
        if sub_goal_dim == prev_sub_goal:
            consecutive_same_dim += 1
        else:
            consecutive_same_dim = 0
            sub_reward += 0.1  # Bonus for dimension switch
        prev_sub_goal = sub_goal_dim
        meta_state = state
        for h in range(5):
            if done or steps >= 200:
                break
            action_idx = agent.choose_action(state, sub_goal_dim)
            action = env.actions[action_idx]
            dim = action[0] # Extract dim from action for per-dim modulation
            next_state, reward, done = env.step(action)
            agent.train_step(state, action_idx, reward, next_state, done, steps, causal_intervention, dim)
            sub_reward += reward
            total_reward += reward
            state = next_state
            steps += 1
        if consecutive_same_dim > 2:
            sub_reward -= 0.05 * consecutive_same_dim  # Penalty for stuck in same dim
        meta_controller.store_transition(meta_state, sub_goal_dim, sub_reward, state, done)
        meta_controller.train()
    print(f"Episode {episode} ended, Reward: {total_reward}, Steps: {steps}, Done: {done}, Agent Epsilon: {agent.epsilon:.3f}, Meta Epsilon: {meta_controller.epsilon:.3f}")
    rewards.append(total_reward)
    recent_rewards.append(total_reward)
    agent.update_epsilon()
    meta_controller.update_epsilon()
    if episode % 5 == 0:
        agent.update_target()
        meta_controller.update_target()
    agent.check_dynamic_growth(recent_rewards, optimal_reward)
    progress = curriculum.check_progress(recent_rewards, optimal_reward)
    print(f"Recent rewards mean: {np.mean(recent_rewards):.2f}, Optimal: {optimal_reward}, Progress to next dim: {progress}")
    if progress:
        memory.memory.clear() # Reset for new dim
        memory.priorities.clear()
    memory.consolidate(agent) # Episodic Memory Consolidation
    gc.collect() # Clear memory after episode
# Output results
print("Rewards:", rewards)
print(f"Average reward: {np.mean(rewards):.2f}, Max: {max(rewards)}, Min: {min(rewards)}")
input("Press Enter to exit...")