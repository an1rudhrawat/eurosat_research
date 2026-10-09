import common
import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np
import csv
import os

common.set_seed()

train_data = datasets.ImageFolder(
    common.DATA_DIR,
    transform = common.train_tf
)

eval_data = datasets.ImageFolder(
    common.DATA_DIR,
    transform = common.test_tf
)

train_indices, val_indices, test_indices = common.get_split(train_data)

train_dataset = Subset(train_data, train_indices)
val_dataset = Subset(eval_data, val_indices)
test_dataset = Subset(eval_data, test_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

val_loader = DataLoader(
    val_dataset,
    batch_size = 64,
    shuffle = False
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = False
)

model = common.get_model()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.0003
)


num_epochs = 5
best_val_accuracy = 0.0
history = []

for epoch in range(num_epochs):
    # Training phase 
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(common.device), labels.to(common.device)

        # Clear gradients from prev batches
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = output.argmax(dim = 1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    # Validation phase 
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(common.device), labels.to(common.device)

            output = model(images)
            loss = criterion(output, labels)

            val_loss += loss.item() * images.size(0)
            predictions = output.argmax(dim = 1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_accuracy = val_correct / val_total

    # Printing the results for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    # Save the model if validation accuracy improves
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            model.state_dict(),
            "best_model.pth"
        )
        print("Model saved.")


# Test Phase

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_labels = []
all_predictions = []
all_confidences = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(common.device), labels.to(common.device)

        output = model(images)
        probabilites = torch.softmax(output, dim = 1)
        confidence, predictions = probabilites.max(dim = 1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_confidences.extend(confidence.cpu().tolist())

test_accuracy = float(np.mean(np.array(all_labels) == np.array(all_predictions)))
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion matrix, rows -> true labels, columns -> predicted labels
confusion = np.zeros((10, 10), dtype = int)
for true, pred in zip(all_labels, all_predictions):
    confusion[true][pred] += 1

# Accuract for each class
class_names = eval_data.classes
per_class_accuracy = {}
for i, class_name in enumerate(class_names):
    per_class_accuracy[class_name] = float(confusion[i][i] / confusion[i].sum())
    print(f"{class_name}: {per_class_accuracy[class_name]:.4f}")

# Saving everything
os.makedirs("results/baseline", exist_ok = True)
np.save("results/baseline/confusion_matrix.npy", confusion)
with open("results/baseline/per_class_accuracy.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset_indx", "true_label", "predicted_label", "confidence"])
    for i in range(len(test_indices)):
        writer.writerow([
            int(test_indices[i]),
            all_labels[i],
            all_predictions[i],
            all_confidences[i]
        ])

metrics = {
    "seed": common.SEED,
    "n_train": len(train_indices),
    "n_val": len(val_indices),
    "n_test": len(test_indices),
    "best_val_accuracy": best_val_accuracy,
    "test_accuracy": test_accuracy,
    "per_class_accuracy": per_class_accuracy,
    "history": history
}

with open("results/baseline/metrics.json", "w") as f:
    json.dump(metrics, f, indent = 2)

split = {
    "train": [int(i) for i in train_indices],
    "val": [int(i) for i in val_indices],
    "test": [int(i) for i in test_indices]
}

with open("results/baseline/split.json", "w") as f:
    json.dump(split, f)

print("Results saved!")