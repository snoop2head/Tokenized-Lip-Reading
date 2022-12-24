import torch


def check_device():
    # device designation
    if torch.cuda.is_available():
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("GPU Name:", torch.cuda.get_device_name(0))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("No GPU, using CPU.")
        device = torch.device("cpu")

    # if cpu then num_workers are 0 else num_workers = 2
    NUM_WORKERS = 12 if torch.cuda.is_available() else 0

    return device, NUM_WORKERS
