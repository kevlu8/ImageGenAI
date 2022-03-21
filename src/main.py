from __future__ import print_function
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

# Set random seed for new images
rseed = random.randint(1, 10000)
print("Random Seed: ", rseed)
random.seed(rseed)
torch.manual_seed(rseed)

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (ngf) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu).to(device)

if os.path.exists('models/generator.pth'):
    netG.load_state_dict(torch.load("models/generator.pth", map_location=device)["model"])
    netG.eval()
else:
    print("No previous model found")
    sys.exit()

# gen = Generator(ngpu).to(device)
# gen.load_state_dict(torch.load("models/generator.pth", map_location=device)["model"])
# gen.eval()

output = []

# Generate images
for i in range(104):
    print("Generating image " + str(i + 1))
    noise = torch.randn(1, nz, 1, 1, device=device)
    gen_img = netG(noise)
    output.append(gen_img.detach().cpu())

# Save the images to output/
output = torch.cat(output)
vutils.save_image(output, "output/gan_output.png", normalize=True)

# Save individual images to output/
for i in range(104):
    vutils.save_image(output[i], "output/gan_output_" + str(i + 1) + ".png", normalize=True)
    