# Bayesian Federated Learning via Predictive Distribution Distillation

This repository provides code for FedPPD. 

FedPPD is a Bayesian federated learning algorithm in which every client learns predictive posterior distribution over its private dataset and share it with the server. The 





![FedDiagram](images/FedDiagram.png)

## Setup

- #### Requirements

  - Python 3.7

  - PyTorch 1.11.0

  - Torchvision 0.12.0

  - Cuda 10.2
  - Tensorboard 2.8.0

- #### Dataset

  - Use torchvision to download MNIST, CIFAR-10, and CIFAR-100 dataset

  - Use [Leaf benchmark](https://github.com/TalwalkarLab/leaf/)  to download FEMNIST dataset and run command 

    ```./preprocess.sh -s iid --iu 0.003 --sf 0.3 -t sample```

    It will generate train and test directory. Split train directory into train and server_data directory for performing distillation at server in *FedPPD+Distill*

- #### Commands

  - MNIST

    ```
    
    ```

    

  





