# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class UnetGenerator_hardware_nndct(torch.nn.Module):
    def __init__(self):
        super(UnetGenerator_hardware_nndct, self).__init__()
        self.module_0 = py_nndct.nn.Input() #UnetGenerator_hardware_nndct::input_0
        self.module_1 = py_nndct.nn.quant_input() #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/QuantStub[quant_stub]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/Conv2d[0]/input.3
        self.module_3 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/LeakyReLU[0]/input.5
        self.module_4 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/Conv2d[1]/input.7
        self.module_5 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/LeakyReLU[0]/input.9
        self.module_6 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/Conv2d[1]/input.11
        self.module_7 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/LeakyReLU[0]/input.13
        self.module_8 = py_nndct.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/Conv2d[1]/input.15
        self.module_9 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/LeakyReLU[0]/input.17
        self.module_10 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/Conv2d[1]/input.19
        self.module_11 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/LeakyReLU[0]/input.21
        self.module_12 = py_nndct.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/Conv2d[1]/2809
        self.module_13 = py_nndct.nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/ConvTranspose2d[2]/input.23
        self.module_14 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/ReLU[3]/2830
        self.module_15 = py_nndct.nn.Cat() #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Cat[Cat]/2836
        self.module_16 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/ConvTranspose2d[3]/input.25
        self.module_17 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/ReLU[4]/2857
        self.module_18 = py_nndct.nn.Cat() #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Cat[Cat]/2863
        self.module_19 = py_nndct.nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/ConvTranspose2d[3]/input.27
        self.module_20 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/ReLU[4]/2884
        self.module_21 = py_nndct.nn.Cat() #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Cat[Cat]/2890
        self.module_22 = py_nndct.nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/ConvTranspose2d[3]/input.29
        self.module_23 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Sequential[model]/ReLU[4]/2911
        self.module_24 = py_nndct.nn.Cat() #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[2]/Cat[Cat]/2917
        self.module_25 = py_nndct.nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/ConvTranspose2d[3]/input.31
        self.module_26 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Sequential[model]/ReLU[4]/2938
        self.module_27 = py_nndct.nn.Cat() #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/UnetSkipConnectionBlock_hardware_nndct[1]/Cat[Cat]/2944
        self.module_28 = py_nndct.nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=True, dilation=[1, 1]) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/ConvTranspose2d[2]/input
        self.module_29 = py_nndct.nn.ReLU(inplace=False) #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/UnetSkipConnectionBlock_hardware_nndct[model]/Sequential[model]/ReLU[3]/inputs
        self.module_30 = py_nndct.nn.dequant_output() #UnetGenerator_hardware_nndct::UnetGenerator_hardware_nndct/DeQuantStub[dequant_stub]/2965

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_4 = self.module_4(output_module_0)
        output_module_4 = self.module_5(output_module_4)
        output_module_6 = self.module_6(output_module_4)
        output_module_6 = self.module_7(output_module_6)
        output_module_8 = self.module_8(output_module_6)
        output_module_8 = self.module_9(output_module_8)
        output_module_10 = self.module_10(output_module_8)
        output_module_10 = self.module_11(output_module_10)
        output_module_12 = self.module_12(output_module_10)
        output_module_12 = self.module_13(output_module_12)
        output_module_12 = self.module_14(output_module_12)
        output_module_15 = self.module_15(dim=1, tensors=[output_module_10,output_module_12])
        output_module_15 = self.module_16(output_module_15)
        output_module_15 = self.module_17(output_module_15)
        output_module_18 = self.module_18(dim=1, tensors=[output_module_8,output_module_15])
        output_module_18 = self.module_19(output_module_18)
        output_module_18 = self.module_20(output_module_18)
        output_module_21 = self.module_21(dim=1, tensors=[output_module_6,output_module_18])
        output_module_21 = self.module_22(output_module_21)
        output_module_21 = self.module_23(output_module_21)
        output_module_24 = self.module_24(dim=1, tensors=[output_module_4,output_module_21])
        output_module_24 = self.module_25(output_module_24)
        output_module_24 = self.module_26(output_module_24)
        output_module_27 = self.module_27(dim=1, tensors=[output_module_0,output_module_24])
        output_module_27 = self.module_28(output_module_27)
        output_module_27 = self.module_29(output_module_27)
        output_module_27 = self.module_30(input=output_module_27)
        return output_module_27
