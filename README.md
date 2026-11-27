# Project-DOOM

**Recurrent Reinforcement Learning for Autonomous DOOM Gameplay**

Yugaantar'26 | DevForge | Track 1: Can It Play DOOM?

## Overview

Project-DOOM is an autonomous reinforcement learning agent designed to play DOOM levels using egocentric observations, recurrent memory and learned neural policies.

The project addresses combat, survival, spatial exploration, keycard acquisition, locked-door traversal and objective-directed navigation without relying on predefined routes or hardcoded gameplay strategies.

## Key Achievements

| Metric | Result |
|---|---:|
| Cumulative training decisions | 709,632 |
| Neural network parameters | 717,706 |
| Procedural keycard maps | 12 |
| Procedural evaluation episodes | 36 |
| Procedural objective completion | 100% |
| Held-out procedural variant completion | 100% |
| Final training throughput | 108.5 decisions/sec |
| Median inference latency | 4.2 ms |
| Peak additional GPU memory | Approximately 140 MB |
| Official local validation | Passed |

Procedural evaluation results apply to the generated map family and do not represent official hidden-map competition scores.

## Unique Selling Proposition

Project-DOOM combines lightweight spatial perception, attention-based entity processing, recurrent memory and objective-specific procedural curriculum learning within a custom reinforcement learning architecture.

Instead of processing computationally expensive raw image frames, the agent learns from structured spatial observations supplied by the official environment.

The procedural curriculum specifically addresses the long-horizon dependency between collecting a keycard, traversing a locked door and reaching an exit.

## Key Features

- Custom neural network with 717,706 parameters.
- Circular convolutional encoding of 360-degree spatial observations.
- Attention-based encoding of visible entities.
- Recurrent memory using a 256-dimensional Gated Recurrent Unit.
- Custom recurrent Proximal Policy Optimization.
- Generalized Advantage Estimation and clipped policy optimization.
- Exploration-driven rewards and count-based curiosity.
- Eight parallel training environments with batched GPU inference.
- Procedural generation of keycard-based training maps.
- Objective-focused reward engineering.
- Independent benchmarking against a random baseline.
- Lightweight neural inference compatible with the official agent interface.

## System Architecture

```text
               Official DOOM Environment
                          |
                Egocentric Observations
                          |
       +------------------+------------------+
       |                  |                  |
   Spatial Rays     Visible Entities       Vitals
       |                  |                  |
   Circular CNN     Entity Attention     Vitals MLP
       |                  |                  |
       +------------------+------------------+
                          |
               Feature Concatenation
                          |
               Previous Action Vector
                          |
                  Feature Fusion
                          |
                  GRU Memory (256)
                          |
               +----------+----------+
               |                     |
          Actor Network        Critic Network
               |                     |
        9 Binary Actions         State Value
               |
        DOOM Environment
```

### Architecture Components

| Component | Function |
|---|---|
| Spatial encoder | Processes 64 directional rays using circular convolutions |
| Entity encoder | Encodes entity types, relative positions and velocities |
| Attention mechanism | Aggregates relevant visible entity information |
| Vitals encoder | Processes health, ammunition, inventory and movement information |
| Feature fusion | Combines encoded observations and previous actions |
| GRU | Maintains recurrent information across successive decisions |
| Actor | Produces probabilities for nine binary actions |
| Critic | Estimates state value during reinforcement learning |

The recurrent state is reset at the beginning of every evaluation episode.

## Training Methodology

Training was conducted across four successive model development phases.

| Version | Training focus | Cumulative decisions |
|---|---|---:|
| V1 | Foundational curriculum learning | 204,800 |
| V2 | Parallel exploration training | 409,600 |
| V3 | Objective-focused fine-tuning | 500,736 |
| V4 | Procedural keycard curriculum | 709,632 |

### Phase 1: Foundational Curriculum Learning

The initial training phase progressively introduces combat, navigation and increasingly complex Freedoom maps.

A custom recurrent PPO implementation optimizes the policy using generalized advantage estimation, clipped policy updates, entropy regularization and recurrent sequence training.

### Phase 2: Exploration-Focused Fine-Tuning

Independent environment workers and batched GPU inference improve training throughput.

Spatial exploration rewards encourage the agent to discover unfamiliar regions rather than repeatedly remaining within limited areas.

Recorded training throughput increased from approximately 17.4 decisions per second near the end of V1 training to 99.7 decisions per second near the end of V2 training.

These measurements were obtained under different training configurations and are not a controlled hardware benchmark.

### Phase 3: Objective-Focused Fine-Tuning

Reward coefficients were revised to emphasize keycard acquisition, door interaction, exit completion and survival.

Training was interrupted after observing limited improvement in keycard acquisition on complex maps. The checkpoint subsequently loaded for V4 contained 500,736 recorded training decisions.

### Phase 4: Procedural Keycard Curriculum

A custom procedural map generator creates 12 maps containing blue, red and yellow keycards, locked doors and exit objectives.

Six procedural variants were included in the specialized training curriculum. Six additional variants were reserved for independent evaluation.

The final training phase emphasizes the following objective sequence:

```text
Locate keycard
      |
Collect keycard
      |
Interact with locked door
      |
Traverse doorway
      |
Reach exit
```

The final checkpoint completed 709,632 cumulative training decisions, with a recorded throughput of 108.5 decisions per second.

Privileged environment information is used during training for reward computation and during local benchmarking for performance measurement. The submitted agent does not access privileged state when selecting actions.

## Reward Engineering

The training pipeline incorporates reward components for:

- Enemy eliminations.
- Damage avoidance and survival.
- Health and item acquisition.
- Spatial exploration.
- Count-based curiosity.
- Keycard acquisition.
- Door interaction.
- Successful exit completion.
- Time efficiency.

Reward coefficients vary across training phases according to their respective learning objectives.

## Experimental Results

### Procedural Objective Benchmark

The procedural benchmark evaluates 12 maps using three fixed seeds per map, resulting in 36 evaluation episodes per checkpoint.

| Metric | V3 | V4 |
|---|---:|---:|
| Keycard acquisition | 91.7% | 100% |
| Door traversal | 91.7% | 100% |
| Overall exit completion | 16.7% | 100% |
| Training variant completion | 5.6% | 100% |
| Held-out variant completion | 27.8% | 100% |

The specialized procedural curriculum increased overall objective completion from 6 of 36 episodes to 36 of 36 episodes.

Door traversal is measured using a position-based criterion specific to the generated map layout.

### General Gameplay Benchmark

The general benchmark uses seven maps and three fixed seeds per map, comparing the trained agent against a random-action baseline.

The evaluation includes Basic, My Way Home, Two Rooms and four held-out Freedoom maps.

| Map | Random exit rate | Final V4 exit rate |
|---|---:|---:|
| Basic | 66.7% | 100% |
| My Way Home | 0% | 33.3% |
| Two Rooms | 0% | 0% |
| Freedoom MAP17 | 0% | 0% |
| Freedoom MAP18 | 0% | 0% |
| Freedoom MAP19 | 0% | 0% |
| Freedoom MAP20 | 0% | 0% |
| Overall | 9.5% | 19.0% |

The final checkpoint demonstrates strong procedural objective execution but weaker survival and combat performance on complex Freedoom maps.

General benchmark results and training statistics are available in `training/logs/`.

### Final Checkpoint Selection

V4 was selected because of its demonstrated ability to complete the full keycard, locked-door and exit sequence.

The earlier checkpoints are retained for comparative evaluation and experimental reproducibility.

The submitted agent loads:

```text
submission/weights/doom_cortex_keys.pt
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.13 | Core implementation |
| PyTorch | Neural networks and reinforcement learning |
| ViZDoom 1.3.3 | Game simulation |
| Gymnasium | Reinforcement learning environment |
| NumPy | Numerical computation |
| NVIDIA CUDA | GPU-accelerated training |
| Python Multiprocessing | Parallel environment workers |
| uv | Python environment management |
| Git LFS | Model checkpoint storage |

## Repository Structure

```text
Project-DOOM/
├── docs/
├── docker/
├── env/
├── maps/
│   ├── two_rooms.wad
│   └── procedural/
├── submission/
│   ├── agent.py
│   ├── model.py
│   ├── requirements.txt
│   └── weights/
│       ├── doom_cortex.pt
│       ├── doom_cortex_parallel.pt
│       ├── doom_cortex_objective.pt
│       └── doom_cortex_keys.pt
├── training/
│   ├── train.py
│   ├── train_parallel.py
│   ├── train_objective.py
│   ├── train_keys.py
│   ├── curriculum.py
│   ├── reward_wrapper.py
│   ├── generate_maps.py
│   ├── benchmark.py
│   ├── benchmark_keys.py
│   ├── check_model.py
│   ├── check_rewards.py
│   ├── check_curriculum.py
│   ├── check_maps.py
│   ├── inspect_maps.py
│   └── logs/
├── tests/
├── validate.py
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Local Setup

Install Python 3.13, uv and Git LFS.

```bash
git clone https://github.com/Chetan941/Project-DOOM.git
cd Project-DOOM
git lfs install
git lfs pull
uv sync
```

Install PyTorch separately. For a compatible NVIDIA GPU with CUDA 12.8 support:

```bash
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA availability:

```bash
uv run --no-sync python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The `--no-sync` option preserves the separately installed PyTorch package.

## Training

The following commands document the training workflow used during development.

### Phase 1: Initial Curriculum Training

```bash
uv run --no-sync python training/train.py --steps 4096 --envs 4 --device cuda

uv run --no-sync python training/train.py --steps 200000 --envs 4 --device cuda --resume
```

### Phase 2: Parallel Exploration Training

```bash
uv run --no-sync python training/train_parallel.py --steps 4096 --envs 8 --rollout 128 --device cuda --resume --seed 123

uv run --no-sync python training/train_parallel.py --steps 200000 --envs 8 --rollout 128 --device cuda --resume --seed 456
```

### Phase 3: Objective-Focused Training

```bash
uv run --no-sync python training/train_objective.py --steps 8192 --envs 8 --rollout 128 --device cuda --resume --seed 789

uv run --no-sync python training/train_objective.py --steps 300000 --envs 8 --rollout 128 --device cuda --resume --seed 1024
```

The second command was interrupted during development. The checkpoint used to initialize Phase 4 contained 500,736 recorded decisions.

### Phase 4: Procedural Keycard Training

Generate the procedural maps:

```bash
uv run --no-sync python training/generate_maps.py
```

Perform procedural map smoke tests:

```bash
uv run --no-sync python training/check_maps.py
```

Train the keycard-focused policy:

```bash
uv run --no-sync python training/train_keys.py --steps 8192 --envs 8 --rollout 128 --device cuda --resume --seed 2048

uv run --no-sync python training/train_keys.py --steps 200000 --envs 8 --rollout 128 --device cuda --resume --seed 4096
```

**Reproducibility note:** Resume commands require the appropriate preceding checkpoints. Back up existing checkpoints before retraining, as training scripts may overwrite their respective output files. Reproducing the historical training sequence requires starting from the correct checkpoint at each phase. Exact numerical results may vary across hardware and execution environments.

## Benchmarking

### General Gameplay Evaluation

```bash
uv run --no-sync python training/benchmark.py
```

Evaluation metrics include:

- Map completion.
- Enemy eliminations.
- Spatial exploration.
- Keycard acquisition.
- Survival and remaining health.
- Damage received.
- Environment reward.
- Inference latency.

### Procedural Keycard Evaluation

```bash
uv run --no-sync python training/benchmark_keys.py
```

This benchmark measures keycard acquisition, door traversal and exit completion across all 12 procedural variants.

Both benchmark scripts evaluate the checkpoint selected in `submission/agent.py`.

## Hardware

Training was conducted on a consumer laptop equipped with:

- AMD Ryzen 7 5800H processor.
- NVIDIA GeForce RTX 3060 Laptop GPU with 6 GB VRAM.
- 16 GB RAM.
- Windows operating system.

## Official Validation

```bash
uv run --no-sync python validate.py
```

The final V4 checkpoint passed the official local validator.

| Validation metric | Measured result | Official limit |
|---|---:|---:|
| Initialization time | 2.6 s | 60 s |
| First-action latency | 286 ms | 10 s |
| Median action latency | 4.2 ms | — |
| 99th percentile latency | 9.1 ms | — |
| Episode agent computation | 2.1 s | 21 s |
| Additional GPU memory | Approximately 140 MB | 2,048 MB |

The validator confirmed the required agent interface and measured resource constraints. The validation episode ended in death after 413 decisions, so validation success should not be interpreted as map completion.

Single-CPU enforcement and peak RAM measurement could not be verified on the local Windows environment.

Docker was unavailable on the development machine, so Linux container validation was not performed.

## Competition Compliance

- Uses the official DevForge environment and agent interface.
- Evaluation actions originate from trained neural parameters.
- No hardcoded pathfinding or scripted combat policies in the submission agent.
- No privileged game-state access during action selection.
- No access to hidden evaluation maps during inference.
- No network access required during inference.
- No repository write access required during inference.
- Model checkpoints managed using Git LFS.
- No additional submission dependencies beyond the official environment.

Scripted procedural map smoke tests are used exclusively for training environment verification and are not part of the submitted gameplay policy.

## Known Limitations

- Generalization to complex unseen maps remains limited.
- The final checkpoint did not complete the four held-out Freedoom maps.
- Specialized keycard training reduced general combat and survival performance relative to earlier checkpoints.
- Held-out procedural variants share the same underlying map family, limiting the scope of the demonstrated generalization.
- Linux container resource compliance has not been independently verified.
- Official hidden-map evaluation results are not yet available.

## Future Improvements

- More diverse procedural map layouts.
- Joint combat and objective curriculum training.
- Improved long-horizon exploration.
- Better retention of previously learned capabilities during specialized fine-tuning.
- Larger independent evaluation suites.

## Official Documentation

- [Agent Contract](docs/CONTRACT.md)
- [Competition Rules](docs/RULES.md)
- [Submission Guide](docs/SUBMISSION.md)
- [Evaluation Details](docs/EVALUATION.md)

---

Developed for Yugaantar'26 DevForge, Track 1: Can It Play DOOM?