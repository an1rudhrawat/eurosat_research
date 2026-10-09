from torchvision.datasets import EuroSAT

def get_dataset():
    dataset = EuroSAT(
        root = "./data",
        download = True
    )
    print(f"Downloaded {len(dataset)} images.")
    return dataset

dataset = get_dataset()

