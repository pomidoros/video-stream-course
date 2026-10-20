"""FlowNetS с явными слоями, совместимая с весами FlowNetPytorch.

Архитектура адаптирована из FlowNetPytorch; MIT-лицензия — vendor/LICENSE.
"""

import torch
from torch import nn


def conv(in_channels, out_channels, kernel_size=3, stride=1):
    """Свёртка и LeakyReLU; stride=2 уменьшает пространственное разрешение."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels,  # Число входных и выходных признаков.
            kernel_size=kernel_size,   # Размер обучаемого пространственного фильтра.
            stride=stride,             # Шаг фильтра по изображению.
            padding=kernel_size // 2,  # При stride=1 сохраняем высоту и ширину.
        ),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),  # Отрицательные значения умножаем на 0.1.
    )


def deconv(in_channels, out_channels):
    """Обучаемое повышение разрешения вдвое."""
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1,  # Размер выхода ровно вдвое больше.
            bias=False,  # Соответствует сохранённым весам модели.
        ),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )


def predict_flow(in_channels):
    """Из признаков получаем два канала смещения: u и v."""
    # Без активации: компоненты потока могут быть отрицательными.
    return nn.Conv2d(in_channels, 2, kernel_size=3, padding=1, bias=False)


def upsample_flow():
    """Увеличить сетку промежуточного потока обучаемым слоем."""
    return nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1, bias=False)


def crop(tensor, reference):
    """Подогнать высоту и ширину перед соединением каналов."""
    # На нечётных размерах encoder и decoder могут различаться на один пиксель.
    return tensor[:, :, :reference.shape[2], :reference.shape[3]]


class SimpleFlowNetS(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder получает два RGB-кадра, соединённых в 6 каналов.
        self.conv1 = conv(6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(256, 256)
        self.conv4 = conv(256, 512, stride=2)
        self.conv4_1 = conv(512, 512)
        self.conv5 = conv(512, 512, stride=2)
        self.conv5_1 = conv(512, 512)
        self.conv6 = conv(512, 1024, stride=2)
        self.conv6_1 = conv(1024, 1024)

        # Decoder соединяет skip-признаки, поднятые признаки и два канала потока.
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)  # 512 + 512 + 2 на предыдущем уровне.
        self.deconv3 = deconv(770, 128)   # 512 + 256 + 2.
        self.deconv2 = deconv(386, 64)    # 256 + 128 + 2.

        # Предсказания на пяти масштабах; flow2 имеет сетку примерно H/4 × W/4.
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)  # 128 + 64 + 2.

        self.upsampled_flow6_to_5 = upsample_flow()
        self.upsampled_flow5_to_4 = upsample_flow()
        self.upsampled_flow4_to_3 = upsample_flow()
        self.upsampled_flow3_to_2 = upsample_flow()

    def forward(self, pair):
        # Сохраняем признаки encoder для skip connections.
        conv2 = self.conv2(self.conv1(pair))
        conv3 = self.conv3_1(self.conv3(conv2))
        conv4 = self.conv4_1(self.conv4(conv3))
        conv5 = self.conv5_1(self.conv5(conv4))
        conv6 = self.conv6_1(self.conv6(conv5))

        # Первое предсказание строится на самой грубой сетке H/64 × W/64.
        flow6 = self.predict_flow6(conv6)
        features5 = crop(self.deconv5(conv6), conv5)
        flow6_up = crop(self.upsampled_flow6_to_5(flow6), conv5)
        concat5 = torch.cat([conv5, features5, flow6_up], dim=1)  # Соединяем каналы.
        flow5 = self.predict_flow5(concat5)

        # Каждый следующий шаг использует encoder-признаки своего масштаба.
        features4 = crop(self.deconv4(concat5), conv4)
        flow5_up = crop(self.upsampled_flow5_to_4(flow5), conv4)
        concat4 = torch.cat([conv4, features4, flow5_up], dim=1)
        flow4 = self.predict_flow4(concat4)

        features3 = crop(self.deconv3(concat4), conv3)
        flow4_up = crop(self.upsampled_flow4_to_3(flow4), conv3)
        concat3 = torch.cat([conv3, features3, flow4_up], dim=1)
        flow3 = self.predict_flow3(concat3)

        features2 = crop(self.deconv2(concat3), conv2)
        flow3_up = crop(self.upsampled_flow3_to_2(flow3), conv2)
        concat2 = torch.cat([conv2, features2, flow3_up], dim=1)
        flow2 = self.predict_flow2(concat2)

        # Несколько выходов нужны для multiscale loss; на инференсе берём flow2.
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        return flow2
