import os
import csv
import time
import psutil
import torch
import torch.distributed as dist
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from dotenv import load_dotenv


def setup_distributed():
    load_dotenv()  # Load configuration from .env
    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', '12345')
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    node_rank = int(os.getenv('NODE_RANK', '0'))
    init_method = f'tcp://{master_addr}:{master_port}'
    dist.init_process_group(backend='gloo', init_method=init_method, world_size=world_size, rank=node_rank)
    return node_rank


def main():
    node_rank = setup_distributed()

    # Transformations for ImageNet images (SqueezeNet input size)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Use the NFS-mounted directory for the ImageNet validation set (SqueezeNet dataset)
    data_dir = os.getenv('DATA_DIR', '/mnt/imagenet')
    dataset = datasets.ImageNet(root=data_dir, split='val', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load pretrained SqueezeNet (trained on ImageNet)
    model = models.squeezenet1_1(pretrained=True)
    model.eval()

    # Count total parameters
    num_params = sum(p.numel() for p in model.parameters())

    # CSV file for appending metrics, configurable via .env
    csv_file = os.getenv('CSV_FILE', f'metrics_{node_rank}.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['timestamp', 'inference_time', 'network_latency', 'accuracy', 'num_parameters', 'cpu_usage',
                 'memory_usage', 'gpu_usage'])

    correct = 0
    total = 0

    for inputs, labels in dataloader:
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(inputs)
        inference_time = time.time() - start_inference

        # Compute accuracy (top-1)
        _, predicted = torch.max(outputs, 1)
        is_correct = int(predicted.item() == labels.item())
        correct += is_correct
        total += 1
        accuracy = correct / total

        # Measure network latency via barrier synchronization
        start_barrier = time.time()
        dist.barrier()
        network_latency = time.time() - start_barrier

        # Collect system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        gpu_usage = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, inference_time, network_latency, accuracy, num_params, cpu_usage, memory_usage, gpu_usage]
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(
            f"Sample {total}: Inference {inference_time:.4f}s, Latency {network_latency:.4f}s, Accuracy {accuracy:.4f}")


if __name__ == '__main__':
    main()
