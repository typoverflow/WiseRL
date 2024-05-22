# Hindsight Preference Learning for Offline Preference-based Reinforcement Learning

## Installation

+ clone this repo and install the dependencies
  ```bash
  git clone https://github.com/typoverflow/WiseRL.git
  cd WiseRL && pip install -e .
  ```
+ install environment or dataset dependencies
  + for D4RL experiments:
    ```bash
    git clone https://github.com/Farama-Foundation/d4rl.git
    cd d4rl
    pip install -e .
    ```
  + for metaworld experiments:
    ```bash
    git clone https://github.com/Farama-Foundation/Metaworld.git
    cd Metaworld && git checkout 04be337a
    pip install -e .
    ```
+ download datasets
  + clone [IPL repo with MIT license](https://github.com/jhejna/inverse-preference-learning?tab=readme-ov-file)
    ```bash
    git clone https://github.com/jhejna/inverse-preference-learning.git
    ```
  + copy the `inverse-preference-learning/datasets/` directory of IPL repo into `WiseRL/datasets/` directory of HPL repo


## Usage

### Gym-Mujoco

Modify the configuration file `scripts/configs/hpl/gym.yaml` and run

```bash
python3 scripts/rmb_main.py --config scripts/configs/hpl/gym.yaml
```

### Adroit

Modify the configuration file `scripts/configs/hpl/adroit.yaml` and run

```bash
python3 scripts/rmb_main.py --config scripts/configs/hpl/adroit.yaml
```

### Metaworld

Modify the configuration file `scripts/configs/hpl/metaworld.yaml` and run

```bash
python3 scripts/rmb_main.py --config scripts/configs/hpl/metaworld.yaml
```
