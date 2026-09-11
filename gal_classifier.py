# KAI STRAKA 09.2026
import GalModel
from GalModel import GalCNN, GalDataset, prep_img

import torch
import torch.nn as nn
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader
from datetime import datetime

id_label = ["Round Elliptical", "In-between Elliptical", "Cigar-shaped Elliptical", \
            "Edge-on Spiral", "Barred Spiral", "Unbarred Spiral", "Irregular", "Merger"]

gal_dataset = load_dataset("mrJordi0/galaxy-zoo-dataset")        # load dataset
test_data = GalDataset(gal_dataset["test"])                      # load test subset
image = Image.open(r"C:\Users\Kai\Desktop\galML\images_training_rev1\133499.jpg")    # call image jpg
image = prep_img(image)
image = image.unsqueeze(0)      # change dimension

model = GalCNN()        # call model

model.load_state_dict(torch.load("galaxy_model1.pth"))      # call saved trained model
criterion = nn.CrossEntropyLoss()                           # loss function

model.eval()            # set evaluation mode

with torch.no_grad():
    output = model(image)                   # logits tensor (0 -> 8)
    probs = torch.softmax(output, 1)        # logits to prob. distribiution
    _, predicted = torch.max(output, 1)     # find highest prob. type
    predict_type = predicted.item()         # find type index in list

print("\nPredicted type: ", id_label[predict_type])
print(f"{probs[0][predict_type] * 100:.2f}%\n")
