import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from torchvision.utils import make_grid, save_image
from PIL import Image

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'cpu'
IMAGE_SIZE = 512


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = torch.flatten(x, 1)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Initial convolution block
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # Residual blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # Upsampling
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 nn.ReflectionPad2d(1),
                                 nn.Conv2d(in_channels, in_channels, 3),
                                 nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.res(x)


class CustomError(Exception):
    pass


def run_transform(image, type_of_transform):
    """
    Превращение лошади(зебры) в зебру(лошадь)
    """
    model = Generator().to(DEVICE)
    # Загружаем веса
    if type_of_transform == 'h2z':
        model.load_state_dict(torch.load(r'GAN_weights/netG_A2B.pth', map_location=DEVICE))
    elif type_of_transform == 'z2h':
        model.load_state_dict(torch.load(r'GAN_weights/netG_B2A.pth', map_location=DEVICE))
    else:
        raise CustomError('type_of_transform is "h2z" or "z2h" only')
    model.eval()

    pre_process = tt.Compose([tt.Resize(IMAGE_SIZE),
                              tt.ToTensor(),
                              tt.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    image = pre_process(image).unsqueeze(0) # Подготавливаем изображние
    image = image.to(DEVICE)
    fake_image = model(image)  # Выход модели
    # Нормализуем изображение (иначе оно остается очень темным)
    grid = make_grid(fake_image.detach(), normalize=True)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0) .to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr) # Получаем PIL-изображение

    return im
