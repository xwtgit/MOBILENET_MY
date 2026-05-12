import copy
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn


def _make_divisible(v: float, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation1d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Hardsigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv1d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return x * self.scale_activation(scale)


class Conv1dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm_layer: Callable[..., nn.Module] | None = nn.BatchNorm1d,
        activation_layer: Callable[..., nn.Module] | None = nn.ReLU,
        dilation: int = 1,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: list[nn.Module] = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=norm_layer is None,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)


@dataclass
class InvertedResidualConfig1d:
    input_channels: int
    kernel: int
    expanded_channels: int
    out_channels: int
    use_se: bool
    activation: str
    stride: int
    dilation: int
    width_mult: float

    def __post_init__(self) -> None:
        self.input_channels = self.adjust_channels(self.input_channels, self.width_mult)
        self.expanded_channels = self.adjust_channels(self.expanded_channels, self.width_mult)
        self.out_channels = self.adjust_channels(self.out_channels, self.width_mult)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float) -> int:
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual1d(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig1d,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation1d, scale_activation=nn.Hardsigmoid),
    ) -> None:
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        activation_layer = nn.Hardswish if cnf.activation == "HS" else nn.ReLU
        layers: list[nn.Module] = []

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv1dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv1dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        layers.append(
            Conv1dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3_1D(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[InvertedResidualConfig1d],
        last_channel: int,
        in_channels: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError("inverted_residual_setting should not be empty")

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers: list[nn.Module] = [
            Conv1dNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        ]

        for cnf in inverted_residual_setting:
            layers.append(InvertedResidual1d(cnf, norm_layer))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            Conv1dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _mobilenet_v3_small_1d_conf(width_mult: float = 1.0, reduced_tail: bool = False) -> tuple[list[InvertedResidualConfig1d], int]:
    reduce_divider = 2 if reduced_tail else 1
    bneck_conf = partial(InvertedResidualConfig1d, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig1d.adjust_channels, width_mult=width_mult)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),
        bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, 1),
    ]
    last_channel = adjust_channels(1024 // reduce_divider)
    return inverted_residual_setting, last_channel


def mobilenet_v3_small_1d(
    in_channels: int = 2,
    num_classes: int = 4,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    reduced_tail: bool = False,
) -> MobileNetV3_1D:
    inverted_residual_setting, last_channel = _mobilenet_v3_small_1d_conf(width_mult, reduced_tail)
    return MobileNetV3_1D(
        inverted_residual_setting,
        last_channel,
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )


def stochastic_depth(input: Tensor, p: float, training: bool) -> Tensor:
    if p == 0.0 or not training:
        return input
    survival_rate = 1.0 - p
    noise = torch.empty([input.shape[0]] + [1] * (input.ndim - 1), dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


class StochasticDepth(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.training)


@dataclass
class _MBConvConfig1d:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: int | None = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig1d(_MBConvConfig1d):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Callable[..., nn.Module] | None = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = int(math.ceil(num_layers * depth_mult))
        if block is None:
            block = MBConv1d
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


class FusedMBConvConfig1d(_MBConvConfig1d):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Callable[..., nn.Module] | None = None,
    ) -> None:
        if block is None:
            block = FusedMBConv1d
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


class MBConv1d(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig1d,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation1d, scale_activation=nn.Sigmoid),
    ) -> None:
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: list[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv1dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        layers.append(
            Conv1dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        layers.append(
            Conv1dNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv1d(nn.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig1d,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        layers: list[nn.Module] = []
        activation_layer = nn.SiLU
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv1dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
            layers.append(
                Conv1dNormActivation(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
        else:
            layers.append(
                Conv1dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet1D(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[_MBConvConfig1d],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        in_channels: int = 2,
        num_classes: int = 4,
        norm_layer: Callable[..., nn.Module] | None = None,
        last_channel: int | None = None,
    ) -> None:
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError("inverted_residual_setting should not be empty")

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers: list[nn.Module] = [
            Conv1dNormActivation(
                in_channels,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        ]

        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: list[nn.Module] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv1dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def efficientnet_v2_s_1d(
    in_channels: int = 2,
    num_classes: int = 4,
    dropout: float = 0.2,
    stochastic_depth_prob: float = 0.2,
) -> EfficientNet1D:
    inverted_residual_setting = [
        FusedMBConvConfig1d(1, 3, 1, 24, 24, 2),
        FusedMBConvConfig1d(4, 3, 2, 24, 48, 4),
        FusedMBConvConfig1d(4, 3, 2, 48, 64, 4),
        MBConvConfig1d(4, 3, 2, 64, 128, 6),
        MBConvConfig1d(6, 3, 1, 128, 160, 9),
        MBConvConfig1d(6, 3, 2, 160, 256, 15),
    ]
    return EfficientNet1D(
        inverted_residual_setting,
        dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        in_channels=in_channels,
        num_classes=num_classes,
        last_channel=1280,
    )


def conv3x3_1d(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1_1d(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck1d(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1_1d(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_1d(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1_1d(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(
        self,
        block: type[Bottleneck1d],
        layers: list[int],
        in_channels: int = 2,
        num_classes: int = 4,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element list")

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1d) and m.bn3.weight is not None:
                    nn.init.zeros_(m.bn3.weight)

    def _make_layer(
        self,
        block: type[Bottleneck1d],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_1d(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet50_1d(
    in_channels: int = 2,
    num_classes: int = 4,
    zero_init_residual: bool = False,
) -> ResNet1D:
    return ResNet1D(
        Bottleneck1d,
        [3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
    )


class _DenseLayer1d(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = norm_layer(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

    def forward(self, input: Tensor) -> Tensor:
        new_features = self.conv1(self.relu1(self.norm1(input)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = nn.functional.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock1d(nn.ModuleDict):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer1d(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                norm_layer=norm_layer,
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for _, layer in self.items():
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition1d(nn.Sequential):
    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            OrderedDict(
                [
                    ("norm", norm_layer(num_input_features)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("conv", nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)),
                    ("pool", nn.AvgPool1d(kernel_size=2, stride=2)),
                ]
            )
        )


class DenseNet1D(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple[int, int, int, int] = (6, 12, 32, 32),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        in_channels: int = 2,
        num_classes: int = 4,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv1d(
                            in_channels,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", norm_layer(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock1d(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                norm_layer=norm_layer,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                transition = _Transition1d(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    norm_layer=norm_layer,
                )
                self.features.add_module(f"transition{i + 1}", transition)
                num_features = num_features // 2

        self.features.add_module("norm5", norm_layer(num_features))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)


def densenet169_1d(
    in_channels: int = 2,
    num_classes: int = 4,
    drop_rate: float = 0.0,
) -> DenseNet1D:
    return DenseNet1D(
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        num_init_features=64,
        bn_size=4,
        drop_rate=drop_rate,
        in_channels=in_channels,
        num_classes=num_classes,
    )


def build_model(
    name: str,
    in_channels: int = 2,
    num_classes: int = 4,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    reduced_tail: bool = False,
    stochastic_depth_prob: float = 0.2,
) -> nn.Module:
    if name == "mobilenet_v3_small_1d":
        return mobilenet_v3_small_1d(
            in_channels=in_channels,
            num_classes=num_classes,
            width_mult=width_mult,
            dropout=dropout,
            reduced_tail=reduced_tail,
        )
    if name == "efficientnet_v2_s_1d":
        return efficientnet_v2_s_1d(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
    )
    if name == "resnet50_1d":
        return resnet50_1d(in_channels=in_channels, num_classes=num_classes)
    if name == "densenet169_1d":
        return densenet169_1d(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")
