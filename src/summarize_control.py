import csv
import json
from collections import Counter

with open("results/baseline/control_review.csv") as f:
    sheet = list(csv.DictReader(f))

with open("results/baseline/control_key.csv") as f:
    key = {r["review_id"]: r["source"] for r in csv.DictReader(f)}

allowed = ["clear", "ambiguous", "label_wrong"]
missing = [r["review_id"] for r in sheet if r["my_review"] not in allowed]
if len(missing) > 0:
    print("These review_ids are empty or misspelled:", missing)
    raise SystemExit

counts = {"error": Counter(), "correct": Counter()}
for r in sheet:
    counts[key[r["review_id"]]][r["my_review"]] += 1

summary = {}
for source in ["error", "correct"]:
    n = sum(counts[source].values())
    unclear = counts[source]["ambiguous"] + counts[source]["label_wrong"]
    percent_unclear = round(100 * unclear / n, 1)
    print(f"{source}: n = {n}, {dict(counts[source])}, unclear (ambiguous + label_wrong) = {percent_unclear}%")
    summary[source] = {
        "n": n,
        "counts": dict(counts[source]),
        "percent_unclear": percent_unclear
    }

with open("results/baseline/control_summary.json", "w") as f:
    json.dump(summary, f, indent = 2)

print("Saved results/baseline/control_summary.json")