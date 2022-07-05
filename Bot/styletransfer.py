import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tt
import torchvision.models as models
import copy

# IMAGE_SIZE = 512 if torch.cuda.is_available() else 128
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_SIZE = 128
DEVICE = 'cpu'
CNN_MEAN = (0.485, 0.456, 0.406)
CNN_STD = (0.229, 0.224, 0.225)
CNN = models.vgg19()
WEIGHTS = torch.load(r'vgg19_weights/vgg19.pth')
CNN = CNN.features.to(DEVICE).eval()


def image_loader(image, image_size=IMAGE_SIZE, return_size=False):
    """
    Загрузчик  PIL-изображний
    """
    loader = tt.Compose([
            tt.Resize(image_size),
            tt.ToTensor()
    ])
    image = loader(image).unsqueeze(0)
    image = image.to(DEVICE, torch.float)
    if return_size:
        return image, image.shape[2:4]  # возвращаем тензор и длину и ширину изображение
    return image


def content_loss(input, target):
    return F.mse_loss(input, target)


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    gram_m = torch.mm(features, features.t())
    gram_m = gram_m.div(a * b * c * d)
    return gram_m


def style_loss(input, target):
    input = gram_matrix(input)
    target = gram_matrix(target)
    return F.mse_loss(target, input)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(DEVICE)
        self.std = torch.tensor(std).view(-1, 1, 1).to(DEVICE)

    def forward(self, input):
        input = (input - self.mean) / self.std
        return input


def get_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer


class VGG(nn.Module):
    """
    Модель на основе VGG19
    """
    def __init__(self, cnn, cnn_mean, cnn_std):
        super(VGG, self).__init__()
        self.features = ['0', '5', '10', '19', '28']  # список номеров слоев, на которых будут считаться лоссы
        last_element = int(self.features[-1])
        norm = Normalization(cnn_mean, cnn_std)
        seq = [norm]
        seq.extend(list(cnn[:last_element + 1].children()))

        self.model = nn.Sequential(*seq)

    def forward(self, x):
        # Store relevant features
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.features:  # Если слой находится в списке,
                features.append(x)               # то запоминаем выход слоя для расчета лосса

        return features


def run_style_transfer(content,
                       style,
                       input,
                       cnn=CNN,
                       cnn_mean=CNN_MEAN,
                       cnn_std=CNN_STD,
                       weights=WEIGHTS,
                       num_steps=250,
                       style_weight=1000000, content_weight=1
                       ):
    """
    Инференс модели
    """
    input = copy.deepcopy(input)
    model = VGG(cnn, cnn_mean, cnn_std)
    model.load_state_dict(weights)  # Загружаем веса модели
    model.requires_grad_(False)  # Замораживаем веса модели
    input.requires_grad_(True)  # Оптимизируем изображение

    optimizer = get_optimizer(input)

    content_features = model(content)
    style_features = model(style)

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input.clamp_(0, 1)

            optimizer.zero_grad()

            style_score = 0
            content_score = 0

            generated_features = model(input)

            for gen_features, con_features, st_features in zip(
                    generated_features,
                    content_features,
                    style_features
            ):
                content_score += content_loss(con_features, gen_features)
                style_score += style_loss(st_features, gen_features)

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            # if run[0] % 50 == 0:
            #     print(f'style_loss: {style_score.item():.4f}, content_loss: {content_score.item():.4f}')
            return style_score + content_score

        optimizer.step(closure)

        with torch.no_grad():
            input.clamp_(0, 1)

    return input  # возвращаем тензор
