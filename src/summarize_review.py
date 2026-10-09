import csv
import json
from collections import Counter

with open("results/baseline/annualcrop_review.csv") as f:
    rows = list(csv.DictReader(f))

allowed = ["label_wrong", "ambiguous", "model_error"]

# Ensure every row is filled
missing = [r["review_id"] for r in rows if r["my_review"] not in allowed]
if len(missing) > 0:
    print("These review_ids are empty or wrongly filled")
    raise SystemExit

n = len(rows)
overall = Counter(r["my_review"] for r in rows)

print("Reviewed", n, "mistakes predicted as AnnualCrop")
for category in allowed:
    count = overall[category]
    print(f"{category}: {count} ({100 * count / n:.1f}% )")

not_clear_error = overall["label_wrong"] + overall["ambiguous"]
print(f"Not a clear model error (label_wrong + ambiguous): {not_clear_error} ({100 * not_clear_error / n:.1f}%)")

# Breakdown by class
by_class = {}

for r in rows:
    if r["true_class"] not in by_class:
        by_class[r["true_class"]] = Counter()
    by_class[r["true_class"]][r["my_review"]] += 1

print("\nBy labelled class:")
for class_name, counts in by_class.items():
    print(" ", class_name, dict(counts))

summary = {
    "n_reviewed": n,
    "overall": dict(overall),
    "percent_not_clear_model_error": round(100 * not_clear_error / n, 1),
    "by_labelled_class": {c: dict(counts) for c, counts in by_class.items()}
}
with open("results/baseline/review_summary.json", "w") as f:
    json.dump(summary, f, indent = 2)

print("\nSaved results/baseline/review_summary.json")