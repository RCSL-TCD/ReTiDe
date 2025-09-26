# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class UnetGenerator_3stage(torch.nn.Module):
    def __init__(self):
        super(UnetGenerator_3stage, self).__init__()
        self.module_0 = py_nndct.nn.Input() #UnetGenerator_3stage::input_0
        self.module_1 = py_nndct.nn.quant_input() #UnetGenerator_3stage::UnetGenerator_3stage/QuantStub[quant_stub]/input.1
        self.module_2 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc1]/Conv2d[proj]/2702
        self.module_3 = py_nndct.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc1]/Conv2d[conv1]/input.3
        self.module_4 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc1]/LeakyReLU[act1]/input.5
        self.module_5 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc1]/Conv2d[conv2]/input.7
        self.module_6 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc1]/LeakyReLU[act2]/input.9
        self.module_7 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc1]/Conv2d[conv3]/2766
        self.module_8 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc1]/Add[add]/input.11
        self.module_9 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc1]/Conv2d[post_res_conv]/input.13
        self.module_10 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc1]/LeakyReLU[act3]/input.15
        self.module_11 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/DownsampleConv[down1]/Conv2d[down]/input.17
        self.module_12 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc2]/Conv2d[proj]/2830
        self.module_13 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc2]/Conv2d[conv1]/input.19
        self.module_14 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc2]/LeakyReLU[act1]/input.21
        self.module_15 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc2]/Conv2d[conv2]/input.23
        self.module_16 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc2]/LeakyReLU[act2]/input.25
        self.module_17 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc2]/Conv2d[conv3]/2894
        self.module_18 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc2]/Add[add]/input.27
        self.module_19 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc2]/Conv2d[post_res_conv]/input.29
        self.module_20 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc2]/LeakyReLU[act3]/input.31
        self.module_21 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/DownsampleConv[down2]/Conv2d[down]/input.33
        self.module_22 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc3]/Conv2d[proj]/2958
        self.module_23 = py_nndct.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc3]/Conv2d[conv1]/input.35
        self.module_24 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc3]/LeakyReLU[act1]/input.37
        self.module_25 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc3]/Conv2d[conv2]/input.39
        self.module_26 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc3]/LeakyReLU[act2]/input.41
        self.module_27 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc3]/Conv2d[conv3]/3022
        self.module_28 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc3]/Add[add]/input.43
        self.module_29 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc3]/Conv2d[post_res_conv]/input.45
        self.module_30 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[enc3]/LeakyReLU[act3]/input.47
        self.module_31 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/DownsampleConv[down3]/Conv2d[down]/input.49
        self.module_32 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[bottleneck]/Conv2d[conv1]/input.51
        self.module_33 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[bottleneck]/LeakyReLU[act1]/input.53
        self.module_34 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[bottleneck]/Conv2d[conv2]/input.55
        self.module_35 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[bottleneck]/LeakyReLU[act2]/input.57
        self.module_36 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[bottleneck]/Conv2d[conv3]/3131
        self.module_37 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[bottleneck]/Add[add]/input.59
        self.module_38 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[bottleneck]/Conv2d[post_res_conv]/input.61
        self.module_39 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[bottleneck]/LeakyReLU[act3]/3155
        self.module_40 = py_nndct.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator_3stage::UnetGenerator_3stage/UpsampleDeconv[up3]/ConvTranspose2d[up]/input.63
        self.module_41 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ReLU[up_relu3]/3176
        self.module_42 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/Add[add3]/input.65
        self.module_43 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec3]/Conv2d[conv1]/input.67
        self.module_44 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec3]/LeakyReLU[act1]/input.69
        self.module_45 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec3]/Conv2d[conv2]/input.71
        self.module_46 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec3]/LeakyReLU[act2]/input.73
        self.module_47 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec3]/Conv2d[conv3]/3243
        self.module_48 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec3]/Add[add]/input.75
        self.module_49 = py_nndct.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec3]/Conv2d[post_res_conv]/input.77
        self.module_50 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec3]/LeakyReLU[act3]/3267
        self.module_51 = py_nndct.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator_3stage::UnetGenerator_3stage/UpsampleDeconv[up2]/ConvTranspose2d[up]/input.79
        self.module_52 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ReLU[up_relu2]/3288
        self.module_53 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/Add[add2]/input.81
        self.module_54 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec2]/Conv2d[conv1]/input.83
        self.module_55 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec2]/LeakyReLU[act1]/input.85
        self.module_56 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec2]/Conv2d[conv2]/input.87
        self.module_57 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec2]/LeakyReLU[act2]/input.89
        self.module_58 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec2]/Conv2d[conv3]/3355
        self.module_59 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec2]/Add[add]/input.91
        self.module_60 = py_nndct.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec2]/Conv2d[post_res_conv]/input.93
        self.module_61 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec2]/LeakyReLU[act3]/3379
        self.module_62 = py_nndct.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=[4, 4], stride=[2, 2], padding=[1, 1], output_padding=[0, 0], groups=1, bias=False, dilation=[1, 1]) #UnetGenerator_3stage::UnetGenerator_3stage/UpsampleDeconv[up1]/ConvTranspose2d[up]/input.95
        self.module_63 = py_nndct.nn.ReLU(inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ReLU[up_relu1]/3400
        self.module_64 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/Add[add1]/input.97
        self.module_65 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec1]/Conv2d[conv1]/input.99
        self.module_66 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec1]/LeakyReLU[act1]/input.101
        self.module_67 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec1]/Conv2d[conv2]/input.103
        self.module_68 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec1]/LeakyReLU[act2]/input.105
        self.module_69 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec1]/Conv2d[conv3]/3467
        self.module_70 = py_nndct.nn.Add() #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec1]/Add[add]/input.107
        self.module_71 = py_nndct.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=False) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec1]/Conv2d[post_res_conv]/input.109
        self.module_72 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #UnetGenerator_3stage::UnetGenerator_3stage/ConvBlock3[dec1]/LeakyReLU[act3]/input
        self.module_73 = py_nndct.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #UnetGenerator_3stage::UnetGenerator_3stage/Conv2d[out_conv]/inputs
        self.module_74 = py_nndct.nn.dequant_output() #UnetGenerator_3stage::UnetGenerator_3stage/DeQuantStub[dequant_stub]/3511

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(input=output_module_0)
        output_module_2 = self.module_2(output_module_0)
        output_module_3 = self.module_3(output_module_0)
        output_module_3 = self.module_4(output_module_3)
        output_module_3 = self.module_5(output_module_3)
        output_module_3 = self.module_6(output_module_3)
        output_module_3 = self.module_7(output_module_3)
        output_module_3 = self.module_8(input=output_module_3, other=output_module_2, alpha=1)
        output_module_3 = self.module_9(output_module_3)
        output_module_3 = self.module_10(output_module_3)
        output_module_11 = self.module_11(output_module_3)
        output_module_12 = self.module_12(output_module_11)
        output_module_13 = self.module_13(output_module_11)
        output_module_13 = self.module_14(output_module_13)
        output_module_13 = self.module_15(output_module_13)
        output_module_13 = self.module_16(output_module_13)
        output_module_13 = self.module_17(output_module_13)
        output_module_13 = self.module_18(input=output_module_13, other=output_module_12, alpha=1)
        output_module_13 = self.module_19(output_module_13)
        output_module_13 = self.module_20(output_module_13)
        output_module_21 = self.module_21(output_module_13)
        output_module_22 = self.module_22(output_module_21)
        output_module_23 = self.module_23(output_module_21)
        output_module_23 = self.module_24(output_module_23)
        output_module_23 = self.module_25(output_module_23)
        output_module_23 = self.module_26(output_module_23)
        output_module_23 = self.module_27(output_module_23)
        output_module_23 = self.module_28(input=output_module_23, other=output_module_22, alpha=1)
        output_module_23 = self.module_29(output_module_23)
        output_module_23 = self.module_30(output_module_23)
        output_module_31 = self.module_31(output_module_23)
        output_module_32 = self.module_32(output_module_31)
        output_module_32 = self.module_33(output_module_32)
        output_module_32 = self.module_34(output_module_32)
        output_module_32 = self.module_35(output_module_32)
        output_module_32 = self.module_36(output_module_32)
        output_module_32 = self.module_37(input=output_module_32, other=output_module_31, alpha=1)
        output_module_32 = self.module_38(output_module_32)
        output_module_32 = self.module_39(output_module_32)
        output_module_32 = self.module_40(output_module_32)
        output_module_32 = self.module_41(output_module_32)
        output_module_32 = self.module_42(input=output_module_32, other=output_module_23, alpha=1)
        output_module_43 = self.module_43(output_module_32)
        output_module_43 = self.module_44(output_module_43)
        output_module_43 = self.module_45(output_module_43)
        output_module_43 = self.module_46(output_module_43)
        output_module_43 = self.module_47(output_module_43)
        output_module_43 = self.module_48(input=output_module_43, other=output_module_32, alpha=1)
        output_module_43 = self.module_49(output_module_43)
        output_module_43 = self.module_50(output_module_43)
        output_module_43 = self.module_51(output_module_43)
        output_module_43 = self.module_52(output_module_43)
        output_module_43 = self.module_53(input=output_module_43, other=output_module_13, alpha=1)
        output_module_54 = self.module_54(output_module_43)
        output_module_54 = self.module_55(output_module_54)
        output_module_54 = self.module_56(output_module_54)
        output_module_54 = self.module_57(output_module_54)
        output_module_54 = self.module_58(output_module_54)
        output_module_54 = self.module_59(input=output_module_54, other=output_module_43, alpha=1)
        output_module_54 = self.module_60(output_module_54)
        output_module_54 = self.module_61(output_module_54)
        output_module_54 = self.module_62(output_module_54)
        output_module_54 = self.module_63(output_module_54)
        output_module_54 = self.module_64(input=output_module_54, other=output_module_3, alpha=1)
        output_module_65 = self.module_65(output_module_54)
        output_module_65 = self.module_66(output_module_65)
        output_module_65 = self.module_67(output_module_65)
        output_module_65 = self.module_68(output_module_65)
        output_module_65 = self.module_69(output_module_65)
        output_module_65 = self.module_70(input=output_module_65, other=output_module_54, alpha=1)
        output_module_65 = self.module_71(output_module_65)
        output_module_65 = self.module_72(output_module_65)
        output_module_65 = self.module_73(output_module_65)
        output_module_65 = self.module_74(input=output_module_65)
        return output_module_65
