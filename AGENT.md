# 🤖 Brawl Stars RL Bot - Project Context (v2.0)

This document serves as the context state for any AI agent continuing work on this project.

---

## 📌 Project Overview
An autonomous reinforcement learning (RL) agent trained to play Brawl Stars (Solo Showdown) via the BlueStacks emulator.
It combines **YOLOv8** (for precise object detection) with **RecurrentPPO** (from `sb3-contrib`, featuring an LSTM memory layer) to make real-time decisions.

### 🆕 v2.0 Improvements (April 2026)
- ✅ **Extended Observation Space**: 15 → 26 dimensions
- ✅ **Frame Stacking**: 4 consecutive frames for motion understanding
- ✅ **MultiDiscrete Action Space**: More stable than continuous actions
- ✅ **Health Detection**: Monitors player HP percentage
- ✅ **Improved Reward Shaping**: Better survival, stealth, and cube collection rewards
- ✅ **Poison Direction Vector**: AI knows which direction to flee
- ✅ **TensorBoard Integration**: Full training metrics logging
- ✅ **Best Model Tracking**: Automatic saving of best-performing model
- ✅ **Hyperparameter Tuning**: Optimized for stability and convergence
- ✅ **Unit Tests**: Comprehensive test coverage for critical components

---

## 🏗️ Architecture & Files

### Core Files
| File | Description |
|------|-------------|
| `rl_train.py` | Main training script. Initializes RecurrentPPO with LSTM, loads checkpoints, configures TensorBoard, runs training loop with callbacks (auto-save, stats logging, best model tracking) |
| `rl_env.py` | Core Gymnasium environment (`BrawlStarsYoloEnv`). Handles screenshot processing, YOLO inference, reward calculations, frame stacking, health monitoring, poison detection, and state management |
| `constants.py` | Global configurations (window name, model paths, hardware flags, observation/action space sizes) |
| `dashboard.py` | Flask web server (`http://localhost:5000`) providing live visualization of raw view, 84x84 CNN input, and real-time statistics |

### Modules
| File | Description |
|------|-------------|
| `modules/tui.py` | Rich-based Terminal User Interface (TUI) for beautiful console logging with color-coded messages |
| `modules/windowcapture.py` | Fast `win32gui` screen capture in background thread (~60 FPS) |
| `modules/detection.py` | YOLOv8 inference in separate thread with CUDA/OpenVINO support |
| `modules/print.py` | Colored terminal output helper (`bcolors` class) |

### Tests
| File | Description |
|------|-------------|
| `tests/test_rl_env.py` | Unit tests for observation space, action space, rewards, poison detection, health detection |
| `tests/test_integration.py` | Integration tests for observation vector indices, reward shaping, edge cases |

### Documentation
| File | Description |
|------|-------------|
| `README.md` | Main project documentation with architecture, installation, and usage |
| `SETUP_GUIDE.md` | Step-by-step setup and troubleshooting guide |
| `CONTRIBUTING.md` | Guide for contributors (coding standards, PR process, testing) |
| `AGENT.md` | This file - AI agent context |

---

## 🧠 AI Brain (Observation & Action Spaces)

### Observation Space (MultiInput)
The model receives a dictionary containing two inputs:

#### 1. **`image`**: Stacked Frames
- **Shape**: `(12, 84, 84)` — 4 frames × 3 RGB channels
- **Type**: `Box(low=0, high=255, dtype=np.uint8)`
- **Purpose**: CNN processes temporal information to understand motion

#### 2. **`vector`**: Extended Feature Vector (26 dimensions)

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | Damage flag | {0, 1} | 1.0 if took damage this step |
| 1 | Health % | [-1, 1] | Normalized: 1.0=100%, -1.0=0% |
| 2 | Box count | [0, 1] | Normalized: value/10 |
| 3 | Enemy count | [0, 1] | Normalized: value/8 |
| 4-6 | Enemy #1 | [-1, 1] | (dx, dy, distance) - closest |
| 7-9 | Enemy #2 | [-1, 1] | (dx, dy, distance) - second closest |
| 10-12 | Cubebox | [-1, 1] | (dx, dy, distance) - closest |
| 13-15 | Bush | [-1, 1] | (dx, dy, distance) - closest |
| 16-19 | Walls | {0, 1} | Obstacle presence in W, S, A, D directions |
| 20 | Poison flag | {0, 1} | 1.0 if in poison gas |
| 21-22 | Poison dir | [-1, 1] | (vec_x, vec_y) - normalized direction to flee |
| 24 | Match time | [0, 1] | Normalized duration (max 180s) |
| 25 | Avg enemy dist | [0, 1] | Average distance to all visible enemies |

**Note**: Indices 23 is unused (reserved for future features)

### Action Space
**Type**: `MultiDiscrete([3, 3, 2, 2, 2])` — 72 possible combinations

| Index | Action | Values | Mapped to |
|-------|--------|--------|-----------|
| 0 | Move X | 0=Left, 1=None, 2=Right | A / D |
| 1 | Move Y | 0=Up, 1=None, 2=Down | W / S |
| 2 | Attack | 0=None, 1=Attack | Space |
| 3 | Super | 0=None, 1=Use | E |
| 4 | Gadget | 0=None, 1=Use | F |

**Decoding Logic**:
```python
if move_x == 0: target_keys.add('a')
elif move_x == 2: target_keys.add('d')

if move_y == 0: target_keys.add('w')
elif move_y == 2: target_keys.add('s')

if attack == 1: action_trigger["space"] = True
if spr == 1: action_trigger["e"] = True
if gdgt == 1: action_trigger["f"] = True
```

---

## ⚙️ Core Mechanics & Overrides

### Asynchronous Threading Architecture
```
Main Thread: RL loop (step → obs → reward → model) ~15 FPS
    ↓
WindowCapture Thread: Screenshots ~60 FPS
    ↓
Detection Thread: YOLOv8 inference ~15-30 FPS
    ↓
Damage Monitor Thread: Health + Poison check ~60 Hz
    ↓
Input Executor Thread: PyAutoGUI execution ~200 Hz
```

### Key Features

#### 1. **Asynchronous Input (`_input_executor`)**
- `pyautogui` inputs in dedicated background thread
- Prevents blocking from destroying RL loop's FPS
- `py.PAUSE` set to `0.0` for maximum speed
- Cooldowns prevent spam: Space=0.35s, E=5.0s, F=3.0s

#### 2. **Health Monitor (`_get_health_percentage`)**
- Reads player's HP bar pixels in real-time
- Counts green pixels (healthy) vs red pixels (damaged)
- Normalized to [0, 1] range
- Added to observation vector as index [1]

#### 3. **Damage Monitor (`_damage_monitor`)**
- High-frequency (~60Hz) background thread
- Checks red damage flash above player name tag
- More reliable than main RL loop (~15 FPS)

#### 4. **Absolute Poison Avoidance**
- Uses HSV color masking to detect poison gas
- Calculates **direction vector** to flee
- **Physically overrides** AI input if walking into gas
- Applied penalty: `-0.80` per step in poison

#### 5. **Auto-Reload on Idle Disconnect**
- Detects "Idle Disconnect" popup window in `_damage_monitor` thread
- Detection: gray popup background + cyan RELOAD button
- Auto-presses `R` key to reload match (5s cooldown)
- Logs: `[SYSTEM] Idle Disconnect detected! Pressing R...`

#### 5. **Frame Stacking**
- Buffers last 4 frames in `deque`
- Concatenated as `(12, 84, 84)` tensor
- Allows AI to understand motion and timing

#### 6. **Action Smoothing**
- Small penalty `-0.005` when WASD keys change
- Prevents "jittering" behavior
- Encourages straight-line movement

---

## 💰 Reward Function

### Per-Step Rewards
| Event | Reward | Notes |
|-------|--------|-------|
| Survival | `+0.01 * (1 + duration/60)` | Scales with match length |
| Closing to box | `+0.005` | Encourages collection |
| Shooting at targets | `+0.02` | Precision fire |
| Safe enemy distance | `+0.005` | 0.3-0.6 range |
| Stealth (bush + enemy) | `+0.02` | Hiding strategy |

### Per-Step Penalties
| Event | Penalty | Notes |
|-------|---------|-------|
| Movement change | `-0.005` | Anti-jitter |
| Wasting ammo | `-0.003` | No targets in sight |
| Too close to enemy | `-0.01` | Distance < 0.2 |
| Taking damage | `-0.15` | Significant penalty |
| In poison | `-0.80` | **HUGE penalty** - forces avoidance |
| Low HP (<30%) | `-0.05` | Encourages caution |
| Idling (>10 steps) | `-0.05 * (idle-10)` | **Prevents disconnect** |

### Match-End Bonuses
| Event | Bonus | Notes |
|-------|-------|-------|
| Survival time | `min(duration/120, 2.0)` | Max +2.0 |
| Low HP survival | `+0.5` | If HP<50% and duration>60s |
| Cube collection | `boxes * 0.2` | Per box destroyed |

**Example Match-End**:
```
Survived 90s: +0.75
Low HP bonus: +0.50
5 cubes destroyed: +1.00
Total: +2.25
```

---

## 🔄 Match Lifecycle

### Start
1. Environment waits for YOLO to detect **any valid object**
2. Match timer starts
3. Frame buffer initialized

### During Match
1. Every step: observe → action → reward → step
2. Monitor health, poison, enemies, boxes
3. Override AI if walking into poison

### End
Match concludes if:
- **No objects detected** for `4.0` consecutive seconds
- Indicates: death, match end screen, or loading screen

### Reset
1. Clear frame buffer
2. Reset reward helpers
3. Wait for next match detection
4. Increment match counter

---

## 📊 Hyperparameters

### RecurrentPPO Configuration
```python
{
    "policy": "MultiInputLstmPolicy",
    "n_steps": 4096,           # Steps per rollout
    "batch_size": 128,         # Mini-batch size
    "learning_rate": 1e-4,     # Lower for stability
    "gamma": 0.995,            # Discount factor (long-term planning)
    "ent_coef": 0.05,          # Entropy coefficient (exploration)
    "clip_range": 0.2,         # PPO clipping
    "vf_coef": 0.5,            # Value function coefficient
    "max_grad_norm": 0.5,      # Gradient clipping
    "gae_lambda": 0.95,        # GAE smoothing
    "n_epochs": 10,            # Epochs per update
    "lstm_hidden_size": 256,   # LSTM memory size
    "total_timesteps": 2_000_000,
}
```

### Network Architecture
```python
policy_kwargs = dict(
    net_arch=[
        dict(
            vision=[128, 128],  # CNN for image input
            pi=[256, 128],      # Policy head
            vf=[256, 128]       # Value head
        )
    ],
    lstm_hidden_size=256,
)
```

---

## 🛠️ Development Notes

### Adding New Observation Features
1. Update `Constants.vector_size` in `constants.py`
2. Add feature to `_get_obs()` in `rl_env.py`
3. Document in observation table above
4. Add tests in `tests/test_rl_env.py`

### Modifying Reward Function
1. Edit `step()` method in `rl_env.py`
2. Keep rewards in `[-1, +1]` range per step
3. Use match-end bonuses for sparse rewards
4. Test with unit tests

### Changing Action Space
⚠️ **Warning**: Requires retraining from scratch!
1. Modify `self.action_space` in `__init__()`
2. Update action decoding in `step()`
3. Old model checkpoints become **incompatible**

### Debugging Tips
- Check TensorBoard for reward trends
- Watch Web Dashboard for real-time vision
- Read TUI logs for penalty/reward breakdown
- Use `pytest tests/ -v` to verify components

---

## 📁 File Dependencies

```
rl_train.py
├── rl_env.py
│   ├── constants.py
│   ├── modules/windowcapture.py
│   ├── modules/detection.py
│   ├── modules/tui.py
│   └── dashboard.py
└── constants.py

dashboard.py (standalone)
tests/
├── test_rl_env.py
└── test_integration.py
```

---

## 🚨 Breaking Changes History

### v2.0 (April 2026)
- ❌ **Incompatible with v1.0 models**
- Observation: 15 → 26 dimensions
- Action: Continuous → MultiDiscrete
- Image: 3 → 12 channels (frame stacking)
- Must retrain from scratch

---

## 🔮 Future Improvements (Not Implemented)

- [ ] Curriculum learning (easy → hard scenarios)
- [ ] Data augmentation for CNN
- [ ] Vectorized environments (multiple BlueStacks)
- [ ] Attention mechanism instead of LSTM
- [ ] Imitation learning from human replays
- [ ] Self-play for combat strategies
- [ ] Semantic segmentation for poison/clouds

---

*Last updated: April 2026*
*Maintained by: @wwmaxik*
