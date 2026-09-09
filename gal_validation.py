# KAI STRAKA 09.2026
import GalModel
from GalModel import GalCNN
from GalModel import GalDataset

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from datetime import datetime

val_dataset = load_dataset("mrJordi0/galaxy-zoo-dataset")       # load dataset
val_data = GalDataset(val_dataset["validation"])                # load validation subset

val_loader = DataLoader(val_data, batch_size = 32, shuffle = False)

model = GalCNN()

model.load_state_dict(torch.load("galaxy_model1.pth"))
criterion = nn.CrossEntropyLoss()

model.eval()
running_loss = 0.0
correct = 0
total = 0

start_time = datetime.now()

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)                 # outputs list of class predictions
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)   # finds highest class number
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(".")

avg_loss = running_loss / len(val_loader)
accuracy = 100 * correct / total

print("validation complete.")
print("start time: ", start_time)
print("finish time: ", datetime.now())
print(f"average loss : {avg_loss:.4f}")
print(f"accuracy : {accuracy:.2f}")
print(f"correct : {correct} / {total}")
