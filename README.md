# SqueezeNet-Distributed-Pi-Inferencing
A distributed inference framework for Raspberry Pi clusters using a pre-trained SqueezeNet model with PyTorch. It collects metrics (inference time, network latency, accuracy, and resource usage) while accessing an NFS-mounted ImageNet dataset.

## Features
- Distributed inference using PyTorch's distributed package.
- Pre-trained SqueezeNet model (ImageNet).
- Automated metrics collection and CSV logging.
- Environment configuration via a secure `.env` file.
- Centralized dataset access using NFS.

## Prerequisites
- Python 3.x
- PyTorch & Torchvision
- psutil
- python-dotenv
- An NFS server with the ImageNet dataset mounted on each node

## Installation
1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install torch torchvision psutil python-dotenv
   ```
3. Set up your NFS share and mount it on each Raspberry Pi.

## Config
- Create a `.env` file in the project root:
```
MASTER_ADDR=your_master_ip
MASTER_PORT=your_master_port
WORLD_SIZE=number_of_nodes
NODE_RANK=node_rank_for_this_machine
CSV_FILE=metrics.csv
DATA_DIR=/path/to/nfs/imagenet
```

### Note for distributed devices
- Each device needs to have their own `.env` file with the corresponding `WORLD_SIZE`
- Each devices `.env` will have its own unique `NODE_RANK` parameter

## Usage
Run the distributed inference script using `torchrun`:
```bash
torchrun --nnodes=<world_size> --nproc_per_node=1 --node_rank=<node_rank> --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" distributed_inference.py
```

## Refs
- [PyTorch Distributed Docs](https://pytorch.org/docs/stable/distributed.html)
- [TorchVision Datasets](https://pytorch.org/vision/main/datasets.html)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

## Example if `WORLD_SIZE`=11
**Server .env (NODE_RANK=0)**
```bash
MASTER_ADDR=192.168.1.100    # Server's IP
MASTER_PORT=12345
WORLD_SIZE=11
NODE_RANK=0
CSV_FILE=metrics_server.csv
DATA_DIR=/mnt/imagenet
```
**Raspberry Pi .env(for NODE_RANK=1-10)**
```bash
MASTER_ADDR=192.168.1.100    # Server's IP (shared with all nodes)
MASTER_PORT=12345
WORLD_SIZE=11
NODE_RANK=1    # Change to 2,3,...,10 on other Pis
CSV_FILE=metrics_pi_1.csv     # Rename for each Pi if desired
DATA_DIR=/mnt/imagenet
```
**Server Torchrun Command(node_rank=0)**
```bash
torchrun --nnodes=11 --nproc_per_node=1 --node_rank=0 --master_addr=192.168.1.100 --master_port=12345 distributed_inference.py
```

**Raspberry Pi Torchrun Command(node_rank=1)**
```bash
torchrun --nnodes=11 --nproc_per_node=1 --node_rank=1 --master_addr=192.168.1.100 --master_port=12345 distributed_inference.py
```
