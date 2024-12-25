# Paper Code: **Dynamic Optimization of Portfolio Allocation Using Deep Reinforcement Learning**

## 1. About

### Paper URL：

https://arxiv.org/abs/2412.18563

### Training Rewards：

![training_rewards](https://github.com/user-attachments/assets/268f5f36-ffba-4ac6-8c56-e3b7d431f9ae)


### Results：

![OPT_comparison](https://github.com/user-attachments/assets/200c7406-8489-4d1b-81a8-ddb6f618defa)




This trading environment is developed based on [wassname](https://github.com/wassname)'s [rl-portfolio-management implementation](https://github.com/wassname/rl-portfolio-management), with improvements and migration to the Stable-Baselines3 (SB3) reinforcement learning framework. (Note: This code is compatible with SB3 under Python 3.7)



## 2. Required Python Packages

This code is only compatible with Python 3.7 and can be run on Linux or Windows 10/11 operating systems.

Execute the following commands to install the required Python packages:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install stable-baselines3[extra]==1.3.0
pip install notebook
pip install einops
pip install tables
pip install seaborn
pip install tqdm
pip install openpyxl
```



The paper code for portfolio optimization using SB3 with gymnasium support is available at: https://github.com/watermeloncq/drl-dynamic-optimization-portfolio-SB3py39

## 3. Code Execution Steps and Important Notes

### (1) Code Execution Steps

After environment configuration and package installation, clone or download the repository locally to execute the Jupyter notebook files.

Step 1: Process data files by executing "./data/0. load chinese data 1d multindex.ipynb" .

Step 2: Execute the jupyter notebook files ("XX.ipynb") in the project root directory, for example:

- 10+1assets-drl-portfolio-Longing-stableBaseline3-PPO-VGG1-softmax-maxAVGSharpe.ipynb : (This notebook implements the paper's methodology.)
- 12+1assets-drl-portfolio-Longing-stableBaseline3-SAC-ViT-softmax-maxAVGSharpe-patchsize3x3.ipynb: (This notebook requires intensive computational resources.)

<u>The training process often requires multiple complete restarts to achieve satisfactory backtesting performance, even with confirmed convergence of training rewards in each attempt. A single successful outcome may necessitate numerous training attempts from scratch.</u>

### (2) Important Notes（English & Chinese）

In "./data/0. load chinese data 1d multindex.ipynb" , the parameter test_split=0.08 defines the train-test split ratio. 

Critical Note: To prevent runtime errors, ensure that the test period length exceeds the random sampling interval used in training. For instance, when each training episode samples 128 trading days, the test period must contain a minimum of 128 days of data.

（在"./data/0. load chinese data 1d multindex.ipynb"  中，test_split=0.08 这个参数用于划分训练集和测试集。

注意：测试集的区间长度不能小于训练时随机采样的区间长度，否则会报错。例如，如果每个 episode 随机采样 128 个交易日的数据进行训练，那么测试集的区间长度就不能少于 128 天。）

### (3) How to use TensorBoard to Monitor Training Progress

Once the model training has commenced, navigate to the "./runs" directory and open a terminal. Execute the following command to monitor the training process:

```
tensorboard --logdir ppo-vgg1-softmax
```

 ppo-vgg1-softmax refers to the target folder within the runs directory.

## 4. Citation

If you use this code, please cite our paper:

```bibtex
@misc{huang2024dynamicoptimizationportfolioallocation,
      title={Dynamic Optimization of Portfolio Allocation Using Deep Reinforcement Learning}, 
      author={Gang Huang and Xiaohua Zhou and Qingyang Song},
      year={2024},
      eprint={2412.18563},
      archivePrefix={arXiv},
      primaryClass={q-fin.PM},
      url={https://arxiv.org/abs/2412.18563}, 
}
```

## 5.Comparative Analysis Code: Optimization Performance of DRL versus Traditional Optimization Models

URL: https://github.com/watermeloncq/OPT_comparison_for_paper
