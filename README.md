<p align="center">

<img src="assets/banner.png">

</p>

> 🚧 This repo is subject to major API changes 🚧

WiseRL provides unofficial and banchmarked PyTorch implementations for Offline Preference-Based RL algorithms, including:
- Oracle-IQL & Oracle AWAC
- Supervised Finetuning (SFT)
- BT Model + IQL/AWAC (BT-IQL/AWAC)
- Contrastive Prefereing Learning (CPL)
- Inverse Preference Learning + IQL/AWAC (IPL-IQL/AWAC)

# Installation
+ clone this repo and install the dependencies
  ```bash
  git clone git@github.com:typoverflow/WiseRL
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
    git clone git@github.com:Farama-Foundation/Metaworld
    cd Metaworld && git checkout 04be337a
    pip install -e .
    ```
