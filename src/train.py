# Trains the AI model. Requires a data/ folder with the training data.

import os
import json
import cv2
import PIL

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

import numpy as np
from tqdm import tqdm

MAX_HEIGHT, MAX_WIDTH = 1080, 1920

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 601),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

def convertAndScale(filename):
    try:
        img = PIL.Image.open(filename[0])
    except:
        print("Weird.")
        return
    width, height = img.size
    if width > MAX_WIDTH or height > MAX_HEIGHT:
        print("Need to resize")
        scale = max(width, height) / min(MAX_WIDTH, MAX_HEIGHT)
        if width / scale > MAX_WIDTH or height / scale > MAX_HEIGHT:
            scale = max(width, height) / max(MAX_WIDTH, MAX_HEIGHT)
        print(scale)
        img = img.resize((
            math.floor(width / scale),
            math.floor(height / scale)
        ))
        print(img)
        if filename[0] != filename[1]:
            print("Filename is weird")
            img.save(filename[0], "png")
            copyfile(filename[0], filename[0][:-4] + ".png")
        else:
            file = filename[0] + ".png"
            # pil_image = PIL.Image.open(filename[0])
            img.save(file, "png")
            copyfile(file, filename[0][:-4] + ".png")
            os.remove(filename[0])
            filename = (file, filename[0][:-4] + ".png")
    elif filename[0].lower().endswith((".jpg", ".jpeg")):
        print("No resize")
        # Save the png to the temp file
        # pil_image = PIL.Image.open(filename[0])
        # pil_image.save(file, "png")
        file = filename[0] + ".png"
        # pil_image = PIL.Image.open(filename[0])
        img.save(file, "png")
        copyfile(file, filename[0][:-4] + ".png")
        os.remove(filename[0])
        filename = (file, filename[0][:-4] + ".png")
    print(filename)

def fixFiles():
    folder = "data/imgs/hentai/"
    cwd = os.getcwd()
    os.chdir(folder)
    file_list = os.listdir(".")

    for file in file_list:
        if not file.endswith((".png", ".jpg", ".jpeg")):
            os.remove(file)
        elif file.endswith((".jpg", ".jpeg")):
            fullPath = os.path.join(folder, file)
            convertAndScale((fullPath, fullPath))
        else:
            fullPath = os.path.join(folder, file)
            convertAndScale((fullPath, fullPath))
    os.chdir(cwd)
    print("Fixfiles ran")


def main():
    if not os.path.exists("data"):
        print("You don't have a data folder!")
        exit()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    fixFiles()

    dataloader = DataLoader(ImageDataset(), 64, True)

    model = Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

    if os.path.exists("weights.pth"):
        c = torch.load("weights.pth")
        model.load_state_dict(c["model"])
        optimizer.load_state_dict(c["optimizer"])

    model.train()

    criterion = nn.BCELoss()

    epochs = 25

    for epoch in range(epochs):
        for image, label in tqdm(dataloader):
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        model.cpu()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "weights.pth")
        model.to(device)

    model.cpu()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "weights.pth")
    model.to(device)
    model.eval()
    print(model(torch.Tensor(np.zeros((1, 3, 256, 256))).to(device)).cpu())


if __name__ == "__main__":
    main()
    exit()

print("This is meant to be run as a file, not imported as a module.")
