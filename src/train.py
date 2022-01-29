# Trains the AI model. Requires a data/ folder with the training data.

import os
import json
import cv2

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

import numpy as np
from tqdm import tqdm

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


class ImageDataset(Dataset):
    def __init__(self):
        with open(os.path.join("data", "labels.json")) as f:
            self.labels = json.load(f)
        with open("tags.txt") as f:
            self.tags = f.read().split("\n")
        self.imgs = tuple(self.labels.keys())
        self.img_dir = "data/imgs"
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        img_path = os.path.join(self.img_dir, self.imgs[i]) + ".jpg"
        image = transforms.F.to_tensor(cv2.imread(img_path))
        if image.shape != (3, 256, 256):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # label = torch.Tensor(self.target_transform(set(self.labels[os.path.basename(img_path).split(".")[0]])))
        label = torch.Tensor(self.labels[os.path.basename(img_path)[:-4]])
        return image, label
    
    def target_transform(self, labels):
        label = [0.] * len(self.tags)
        for i, t in enumerate(self.tags):
            if t in labels:
                label[i] = 1.
        return label


def main():
    if not os.path.exists("data"):
        print("You don't have a data folder!")
        exit()

    if not os.path.exists("data/labels.json"):
        print("You don't have any labels!")
        exit()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

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
            # let me copyu from wiki real quick 
            # ^ programmer mindset
        model.cpu()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "weights.pth")
        model.to(device)

    model.cpu()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "weights.pth")
    # weights.pth now has proper values
    model.to(device)
    model.eval()
    print(model(torch.Tensor(np.zeros((1, 3, 256, 256))).to(device)).cpu())


if __name__ == "__main__":
    main()
    exit()

print("This is meant to be run as a file, not imported as a module.")
