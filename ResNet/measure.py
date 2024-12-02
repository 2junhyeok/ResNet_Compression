import time

def measure_latency(model, dataloader, device="cpu"):
    """
    모델의 추론 시간을 측정합니다.
    Args:
        model (nn.Module): PyTorch 모델.
        dataloader (DataLoader): 테스트용 데이터 로더.
        device (str): "cpu" 또는 "cuda".
    Returns:
        float: 평균 추론 시간 (초).
    """
    model.to(device)
    model.eval()
    latencies = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            latencies.append(time.time() - start_time)

    avg_latency = sum(latencies) / len(latencies)
    print(f"Average Latency: {avg_latency:.4f} seconds")
    return avg_latency

def measure_memory(model):
    """
    모델의 메모리 사용량을 측정합니다.
    Args:
        model (nn.Module): PyTorch 모델.
    Returns:
        float: 모델 메모리 사용량 (MB).
    """
    total_params = sum(p.numel() for p in model.parameters())
    memory_usage = total_params * 4 / (1024 ** 2)  # Float32 기준 4 bytes
    print(f"Total Parameters: {total_params}")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    return memory_usage
