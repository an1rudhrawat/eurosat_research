import matplotlib.pyplot as plt
from torchvision import datasets
import common

dataset = datasets.ImageFolder(common.DATA_DIR)

image_index = int(input("Enter an image index (0-27000): "))
img, label = dataset[image_index]
plt.imshow(img)
plt.title(f"Class: {dataset.classes[label]}")
plt.axis("off")
plt.show()

