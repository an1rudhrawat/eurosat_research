from torchvision.datasets import ImageFolder
from collections import Counter

dataset = ImageFolder("data/")

distribution = Counter(dataset.targets)

for idx, count in sorted(distribution.items()):
    print(f"{dataset.classes[idx]}: {count}")