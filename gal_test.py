# KAI STRAKA 09.2026
import GalModel
from GalModel import GalCNN
from GalModel import GalDataset

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from datetime import datetime

test_dataset = load_dataset("mrJordi0/galaxy-zoo-dataset")        # load dataset
test_data = GalDataset(test_dataset["test"])                      # load test subset

test_loader = DataLoader(test_data, batch_size = 32, shuffle = False)       # call DataLoader

model = GalCNN()        # call model

model.load_state_dict(torch.load("galaxy_model1.pth"))      # call saved trained model
criterion = nn.CrossEntropyLoss()           # idk

model.eval()            # set evaluation mode
running_loss = 0.0
correct = 0
total = 0

start_time = datetime.now()     # logging start time

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(".")

avg_loss = running_loss / len(test_loader)
accuracy = 100 * correct / total

print("testing complete.")
print("start time: ", start_time)
print("finish time: ", datetime.now())
print(f"average loss : {avg_loss:.4f}")
print(f"accuracy : {accuracy:.2f}")
print(f"correct : {correct} / {total}")
