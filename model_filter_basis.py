import torch.ao.nn.qat as nnqat
from training_functions import *
import torch.nn as nn
import torch.nn.functional as F
import shift.ste as ste
import torch.quantization as quantization
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
from torch.ao.quantization import QConfig, FusedMovingAvgObsFakeQuantize, MovingAverageMinMaxObserver

def find_specific_patterns(model):
    # find BatchNorm, linear, Relu
    patterns = []
    modules = list(model.named_modules())
    for i in range(len(modules) - 1):
        name1, module1 = modules[i]
        name2, module2 = modules[i + 1]
        if isinstance(module1, nn.BatchNorm2d) and isinstance(module2, nn.ReLU):
            patterns.append([name1, name2])

        if isinstance(module1, nn.Linear) and isinstance(module2, nn.ReLU):
            patterns.append([name1, name2])

    return patterns

class conv_shared_filter_basis(nn.Conv2d):
    # grouped conv with shared filter basis
    def __init__(self, in_channels, k_basis, n_basis, kernel_size, share_basis, stride=1):
        super().__init__(in_channels, (in_channels//k_basis)*n_basis, kernel_size,stride,bias=False)
        self.in_channels = in_channels
        self.k_basis=k_basis
        self.n_basis = n_basis
        self.kernel_size =kernel_size
        self.groups = in_channels//k_basis
        self.weight = share_basis

    def forward(self, x):
        weight = self.weight.repeat(self.groups, 1, 1, 1)
        x= F.conv2d(input=x, weight= weight, bias= self.bias, stride=self.stride, padding=self.kernel_size//2, groups=self.groups)
        return x

class Observed_conv_shared_filter_basis(torch.nn.Module):
    # fake quantized module for filter basis
    def __init__(self, conv, qconfig):
        super().__init__()
        self.conv = conv
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()
    def forward(self, x):
        weight = self.conv.weight.repeat(self.conv.groups, 1, 1, 1)
        weight = self.weight_fake_quant(weight)
        result= F.conv2d(input=x, weight=weight, bias=self.conv.bias, stride=self.conv.stride,
                        padding=self.conv.kernel_size // 2, groups=self.conv.groups)
        return result
    def get_quantized_weights(self):
        return self.weight_fake_quant(self.conv.weight).detach().cpu().numpy()

    @classmethod
    def from_float(cls, float_module):
        assert hasattr(float_module, 'qconfig')
        observed = cls(float_module,float_module.qconfig)
        return observed

class conv2_1x1_shift(nn.Module):
    # do linear combination of filter basis with bitwise shift operation
    def __init__(self, in_channels, out_channels, stride, bias=False):
        super().__init__()
        self.shift_EN=False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.weight = nn.Parameter(
            nn.init.kaiming_uniform_(torch.Tensor(out_channels, in_channels , 1, 1)),
            requires_grad=True)
        self.shift_range = (-15, 0)
        self.rounding = 'deterministic'
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.qconfig = QConfig(torch.nn.Identity,
                               torch.nn.Identity)
        print(self.shift_range)
    def get_quantized_weights(self):
        return ste.round_power_of_2(self.weight, self.rounding)
    def forward(self, x):
        if self.shift_EN:
            self.weight.data = ste.clampabs(self.weight.data, 2 ** self.shift_range[0], 2 ** self.shift_range[1])
            filter = ste.round_power_of_2(self.weight, self.rounding)
        else:
            filter =self.weight
        x= F.conv2d(input=x, weight=filter, bias=self.bias, stride=self.stride)
        return x

class Observed_conv2_1x1_shift(nn.Module):
    #fake quantized module for conv2_1x1_shift
    def __init__(self, conv,qconfig):
        super().__init__()
        self.conv = conv
        self.qconfig = qconfig
        self.weight_fake_quant = self.qconfig.weight()
        self.conv.shift_range = (-7, 0)
        print(self.conv.shift_range)
    def forward(self, x):
        self.conv.weight.data = ste.clampabs(self.conv.weight.data, 2 ** self.conv.shift_range[0], 2 ** self.conv.shift_range[1])
        filter = ste.round_power_of_2(self.conv.weight, self.conv.rounding)
        filter = self.weight_fake_quant(filter)
        return F.conv2d(input=x, weight=filter, bias=self.conv.bias, stride=self.conv.stride)
    def get_quantized_weights(self):
        return ste.round_power_of_2(self.conv.weight, self.conv.rounding).detach().cpu().numpy()
    @classmethod
    def from_float(cls, float_module):
        assert hasattr(float_module, 'qconfig')
        observed = cls(float_module, float_module.qconfig)
        return observed

class CombiConv(nn.Module):
    #replace conv2d with this module. k_basis: channel of filter basis; n_basis: number of filter basis;
    #share_basis: shared filter basis; shift: enable bitwise shift for linear combination coefficients
    def __init__(self, in_channels, out_channels,k_basis, n_basis, kernel_size, share_basis,
                 stride=1, bias=True, shift=False):
        super().__init__()
        modules = [conv_shared_filter_basis(in_channels, k_basis, n_basis, kernel_size, share_basis, stride)]
        if shift:
            conv_1x1=conv2_1x1_shift(in_channels* n_basis // k_basis, out_channels, stride=stride, bias=bias)
        else:
            conv_1x1 = nn.Conv2d(in_channels* n_basis // k_basis, out_channels, kernel_size=1, stride=stride, bias=bias)
            #modules.append(nn.BatchNorm2d(in_channels * n_basis // k_basis))
        modules.append(conv_1x1)
        #modules.append(quantization.QuantStub(post_activation_quantizer_config))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)

#fake quantization module mapping
prepare_custom_config_dict = {
    #"float_to_observed_custom_module_class": {
        conv_shared_filter_basis: Observed_conv_shared_filter_basis,
        conv2_1x1_shift: Observed_conv2_1x1_shift,
        nn.Conv2d: nnqat.Conv2d,
        nn.Linear: nnqat.Linear,
        nni.LinearReLU: nniqat.LinearReLU
    #}
}

class VGG16_shared_basis_filter(nn.Module):
    # compressed vgg16. k_basis: channel of filter basis; n_basis: number of filter basis;
    # shift: enable bitwise shift for linear combination coefficients
    def __init__(self, num_class= 100, input_size =32, k_basis=1,n_basis=3, shift=False):
        super().__init__()

        self.input_size = input_size
        self.share_basis = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(n_basis, k_basis, 3, 3)), requires_grad=True)
        self.features = nn.Sequential(
            #nn.Conv2d(3, 64, kernel_size=3, padding=1),
            CombiConv(3, 64, k_basis, n_basis, 3, share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            CombiConv(64, 64, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CombiConv(64, 128, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CombiConv(128, 128, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CombiConv(128, 256, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            CombiConv(256, 256, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            CombiConv(256, 256, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CombiConv(256, 512, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            CombiConv(512, 512, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            CombiConv(512, 512, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            CombiConv(512, 512, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            CombiConv(512, 512, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            CombiConv(512, 512, k_basis,n_basis, 3,share_basis=self.share_basis,shift=shift),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(512*(self.input_size//32)*(self.input_size//32), 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,num_class)
        )
        self.quantize=quantization.QuantStub(QConfig(FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                      quant_min=0,
                                                                      quant_max=255,
                                                                      dtype=torch.quint8,
                                                                      qscheme=torch.per_tensor_affine),torch.nn.Identity))
        self.dequantize=quantization.DeQuantStub()
    def forward(self, x):
        x = self.quantize(x)
        x = self.features(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequantize(x)
        return x