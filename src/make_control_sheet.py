import common
import csv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

data = datasets.ImageFolder(common.DATA_DIR)
class_names = data.classes
annual_crop = class_names.index("AnnualCrop")
focus = [class_names.index("Pasture"), class_names.index("PermanentCrop")]

with open("results/baseline/per_class_accuracy.csv") as f:
    rows = list(csv.DictReader(f))

# splitting images into mistakes and predicted correctly
errors = {c: [] for c in focus}
correct = {c: [] for c in focus}

for r in rows:
    true_label = int(r["true_label"])
    predicted_label = int(r["predicted_label"])
    idx = int(r["dataset_indx"])

    if true_label not in focus:
        continue
    if predicted_label == annual_crop:
        errors[true_label].append(idx)
    elif predicted_label == true_label:
        correct[true_label].append(idx)

# picking 15 of each and shuffling
rng = np.random.RandomState(common.SEED)
n_each = 15
items = []

for c in focus:
    for idx in rng.choice(errors[c], n_each, replace = False):
        items.append({"dataset_idx": int(idx), "true": c, "source": "error"})
    for idx in rng.choice(correct[c], n_each, replace = False):
        items.append({"dataset_idx": int(idx), "true": c, "source": "correct"})
rng.shuffle(items)
print("Tiles to review:", len(items))

# contact sheet only title and its label
per_page = 20
n_pages = (len(items) + per_page - 1) // per_page

for page in range(n_pages):
    batch = items[page * per_page: (page + 1) * per_page]

    fig, axes = plt.subplots(4, 5, figsize = (12, 10))
    for ax in axes.flatten():
        ax.axis("off")

    for k, (ax, item) in enumerate(zip(axes.flatten(), batch)):
        review_id = page * per_page + k + 1
        image, _ = data[item["dataset_idx"]]
        ax.imshow(image)
        ax.set_title(f"#{review_id}  label: {class_names[item['true']]}")

    fig.savefig(
        f"results/baseline/control_page{page + 1}.png",
        bbox_inches = "tight"
    )
    plt.close(fig)  

# Manual Labour part 2
with open("results/baseline/control_review.csv", "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["review_id", "my_review"])
    for i in range(len(items)):
        writer.writerow([i + 1, "", ""])

with open("results/baseline/control_key.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["review_id", "dataset_idx", "true_class", "source"])
    for i, item in enumerate(items):
        writer.writerow([i + 1, item["dataset_idx"], class_names[item["true"]], item["source"]])

print("Saved", n_pages, " int control_review.csv and control_key.csv")

