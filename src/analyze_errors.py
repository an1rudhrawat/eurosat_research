from torchvision import datasets
from common import DATA_DIR
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

data = datasets.ImageFolder(DATA_DIR)
class_names = data.classes

confusion = np.load("results/baseline/confusion_matrix.npy")

with open("results/baseline/per_class_accuracy.csv", "r") as f:
    rows = list(csv.DictReader(f))

# class which gets the most misclassified pairs
pairs = []
for i in range(10):
    for j in range(10):
        if i != j:
            pairs.append((int(confusion[i][j]), class_names[i], class_names[j]))

pairs.sort(reverse = True)

print("Most confused (count, true class, predicted class):")
for pair in pairs[:8]:
    print(pair)

with open("results/baseline/top_confusions.json", "w") as f:
    json.dump(pairs[:8], f, indent = 2)


# confusion matrix

plt.figure(figsize = (10, 10))
plt.imshow(confusion, cmap = "Blues")
plt.xticks(range(10), labels = class_names, rotation = 90)
plt.yticks(range(10), labels = class_names)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.colorbar()
plt.savefig("results/baseline/confusion_matrix.png", bbox_inches = "tight")
plt.close()

# wrong predictions which model was most confident about
errors = [r for r in rows if r["true_label"] != r["predicted_label"]]
errors.sort(key = lambda r: float(r["confidence"]), reverse = True)
print("Total mistakes on the test set:", len(errors))
errors = errors[:24]

fig, axes = plt.subplots(4, 6, figsize = (12, 8))
for ax in axes.flatten():
    ax.axis("off")

for ax, r in zip(axes.flatten(), errors):
    image, _ = data[int(r["dataset_indx"])]
    ax.imshow(image)
    confidence = float(r["confidence"])
    ax.set_title(
        "True: " + class_names[int(r["true_label"])]
        + "\nPred: " + class_names[int(r["predicted_label"])]
        + f"\nConfidence: {confidence:.2f}"
    )

fig.savefig("results/baseline/top_confident_errors.png", bbox_inches = "tight")
print("Saved top confident errors to results/baseline/top_confident_errors.png")