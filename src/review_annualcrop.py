import common
import csv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets


data = datasets.ImageFolder(common.DATA_DIR)
class_names = data.classes
annual_crop = class_names.index("AnnualCrop")

with open("results/baseline/per_class_accuracy.csv") as f:
    rows = list(csv.DictReader(f))

# Keep only mistakes where model says annual crop
errors = []
for r in rows:
    true_label = int(r["true_label"])
    predicted_label = int(r["predicted_label"])

    if predicted_label == annual_crop and true_label != annual_crop:
        errors.append({
            "dataset_idx": int(r["dataset_indx"]),
            "true": true_label,
            "confidence": float(r["confidence"])
        })

# Most confident mistakes first
errors.sort(key = lambda x: x["confidence"], reverse = True)
print("Mistakes predicted as AnnualCrop", len(errors))

per_page = 20
n_pages = (len(errors) + per_page - 1) // per_page

for page in range(n_pages):
    batch = errors[page * per_page: (page + 1) * per_page]

    fig, axes = plt.subplots(4, 5, figsize = (12, 10))
    for ax in axes.flatten():
        ax.axis("off")

    for k, (ax, e) in enumerate(zip(axes.flatten(), batch)):
        review_id = page * per_page + k + 1
        image, _ = data[e["dataset_idx"]]
        ax.imshow(image)
        ax.set_title(
            f"#{review_id} labelled: {class_names[e['true']]}\n {e['confidence']:.2f}"
        )
    fig.savefig(
        f"results/baseline/annualcrop_review_page{page + 1}.png",
        bbox_inches = "tight"
    )
    plt.close(fig)

# Manual Labor
with open("results/baseline/annualcrop_review.csv", "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["review_id", "dataset_idx","true_class", "confidence", "my_review", "notes"])
    for i, e in enumerate(errors):
        writer.writerow([
            i + 1,
            e["dataset_idx"],
            class_names[e["true"]],
            round(e["confidence"], 3),
            "",""
        ])
print("Saved", n_pages)


