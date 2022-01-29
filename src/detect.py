# This file is where the detection of objects inside of an image occurs.

if __name__ == "__main__":
    print("You're running the wrong file. Please run main.py instead.")
    exit()

import torch
from torch import nn
from torchvision.transforms import transforms
import cv2

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

'''
def detect(imagePath):
    tags = [
        "placeholder"
    ]
    model = train.Net()
    model.load_state_dict(torch.load("weights.pth"))
    model.eval()
    return tags[model(PIL.Image.open(imagePath).convert())]
'''

THRESHOLD = 0.5
MAX_TAGS = 20

def detect(imagePath, eventQueue):
    with open("tags.txt") as f:
        tags = f.read().split("\n")
    model = Net()
    model.load_state_dict(torch.load("weights.pth", "cpu")["model"])
    model.eval()
    estimate = model(transforms.F.to_tensor(cv2.resize(cv2.imread(imagePath, cv2.IMREAD_COLOR), (256, 256))).view((1, 3, 256, 256)).float())
    # tags = model(PIL.Image.open(imagePath))
    output = []
    for i, e in enumerate(estimate[0]):
        if e > THRESHOLD:
            output.append((tags[i], round(e.item(), 3) * 100))
    output.sort(key=lambda a: str(a[1]) + a[0], reverse=True)
    print(estimate)
    eventQueue.append(("detect finished", {"tags": output[:MAX_TAGS]}))
