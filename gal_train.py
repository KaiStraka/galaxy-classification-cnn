# KAI STRAKA 09.2026
import GalModel
from GalModel import GalCNN
from GalModel import GalDataset

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from datetime import datetime

train_dataset = load_dataset("mrJordi0/galaxy-zoo-dataset")         # load dataset
train_data = GalDataset(train_dataset["train"])                     # load training subset

train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)

model = GalCNN()

criterion = nn.CrossEntropyLoss()       # loss function 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005, weight_decay = 5e-4)

start_time = datetime.now()
epochs = 16     # number of full cycles
for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(".")

    print(f"Epoch {epoch + 1}: " f"Loss = {running_loss:.4}")

torch.save(model.state_dict(), "galaxy_model1.pth")
print("training complete.")
print("start time: ", start_time)
print("finish time: ", datetime.now())
