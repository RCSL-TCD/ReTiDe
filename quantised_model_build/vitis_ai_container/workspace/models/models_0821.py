import torch
import torch.nn as nn


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        uprelu = nn.ReLU(inplace=True)
        upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.ReLU()]  # Output activation
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv,  uprelu, upconv, ]
        else:
            model = [downrelu, downconv,  submodule, uprelu, upconv, ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs):
        super(UnetGenerator, self).__init__()

        # Construct U-Net structure
        unet_block = UnetSkipConnectionBlock(512, 512, innermost=True)
        for _ in range(num_downs - 5):  # 5 includes innermost + 4 outer layers
            unet_block = UnetSkipConnectionBlock(512, 512, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(256, 512, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(128, 256, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock(64, 128, submodule=unet_block)
        self.model = UnetSkipConnectionBlock(output_nc, 64, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionBlock_hardware(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock_hardware, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.1015625, inplace=True) # Changhong This value is fixed because hardware will compile leakyrelu with a fixed value for efficiency
        uprelu = nn.ReLU(inplace=True)
        upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [upconv, nn.ReLU()]  # Changhong, delete uprelu
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            model = [downrelu, downconv,  upconv , uprelu]
        else:
            model = [downrelu, downconv,  submodule, upconv , uprelu]# Changhong, make uprelu followed with Conv

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class UnetSkipConnectionBlock_hardware_bilinear(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock_hardware_bilinear, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        # Down path
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.1015625, inplace=True)  # fixed slope for hardware
        uprelu   = nn.ReLU(inplace=True)

        # Up path: bilinear upsample -> 3x3 conv to refine features
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        if outermost:
            # in: inner_nc*2  -> out: outer_nc
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, padding=1, bias=True)
            down = [downconv]
            up   = [upsample, upconv, nn.ReLU(inplace=True)]  # match original outermost ReLU after up
            model = down + [submodule] + up

        elif innermost:
            # in: inner_nc     -> out: outer_nc
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, padding=1, bias=False)
            model = [downrelu, downconv, upsample, upconv, uprelu]

        else:
            # in: inner_nc*2  -> out: outer_nc
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, padding=1, bias=False)
            model = [downrelu, downconv, submodule, upsample, upconv, uprelu]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], dim=1)

class UnetGenerator_hardware(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs):
        super(UnetGenerator_hardware, self).__init__()

        # Construct U-Net structure
        unet_block = UnetSkipConnectionBlock_hardware(512, 512, innermost=True)
        for _ in range(num_downs - 5):  # 5 includes innermost + 4 outer layers
            unet_block = UnetSkipConnectionBlock_hardware(512, 512, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock_hardware(256, 512, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock_hardware(128, 256, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock_hardware(64, 128, submodule=unet_block)
        self.model = UnetSkipConnectionBlock_hardware(output_nc, 64, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)



class UnetSkipConnectionBlock_hardware_pixelshuffle(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock_hardware_pixelshuffle, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.1015625, inplace=True)
        uprelu = nn.ReLU(inplace=True)

        # For PixelShuffle, we need to output 4x more channels to upscale by 2 (2x2)
        if outermost:
            upconv = nn.Conv2d(inner_nc * 2, outer_nc * 4, kernel_size=3, padding=1)
            #up = [upconv, nn.PixelShuffle(2), nn.ReLU()]  # Note: replaced ConvTranspose2d
            up = [upconv, nn.ReLU(), nn.PixelShuffle(2)]  # Note: replaced ConvTranspose2d
            down = [downconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.Conv2d(inner_nc, outer_nc * 4, kernel_size=3, padding=1, bias=False)
            # model = [downrelu, downconv, upconv, nn.PixelShuffle(2), uprelu]
            model = [downrelu, downconv, upconv, uprelu ,nn.PixelShuffle(2)]
        else:
            upconv = nn.Conv2d(inner_nc * 2, outer_nc * 4, kernel_size=3, padding=1, bias=False)
           # model = [downrelu, downconv, submodule, upconv, nn.PixelShuffle(2), uprelu]
            model = [downrelu, downconv, submodule, upconv, uprelu, nn.PixelShuffle(2)]



        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator_hardware_pixelshuffle(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs):
        super(UnetGenerator_hardware_pixelshuffle, self).__init__()

        # Construct U-Net structure
        unet_block = UnetSkipConnectionBlock_hardware_pixelshuffle(512, 512, innermost=True)
        for _ in range(num_downs - 5):  # 5 includes innermost + 4 outer layers
            unet_block = UnetSkipConnectionBlock_hardware_pixelshuffle(512, 512, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock_hardware_pixelshuffle(256, 512, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock_hardware_pixelshuffle(128, 256, submodule=unet_block)
        unet_block = UnetSkipConnectionBlock_hardware_pixelshuffle(64, 128, submodule=unet_block)
        self.model = UnetSkipConnectionBlock_hardware_pixelshuffle(output_nc, 64, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)





# class ConvBlock3(nn.Module):
#     def __init__(self, in_ch, out_ch, negative_slope=0.1015625):
#         super().__init__()
#         self.act = nn.LeakyReLU(negative_slope, inplace=True)

#         self.conv1 = nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False)
#         self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
#         self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)

#         # Projection for channel/shape match on the residual branch
#         if in_ch != out_ch:
#             self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
#         else:
#             self.proj = nn.Identity()

#     def forward(self, x):
#         identity = self.proj(x)

#         y = self.act(self.conv1(x))
#         y = self.act(self.conv2(y))
#         y = self.conv3(y)          # no activation before the add

#         y = y + identity           # residual add
#         y = self.act(y)            # post-residual activation
#         return y

class ConvBlock3(nn.Module):
    def __init__(self, in_ch, out_ch, negative_slope=0.1015625):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

        self.conv1 = nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)

        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

        # NEW: 1×1 conv after residual add to keep "conv -> activation" ordering
        self.post_res_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self._init_post_res_identity()

    def _init_post_res_identity(self):
        """Initialize 1×1 conv to identity mapping (no effect at t=0)."""
        w = self.post_res_conv.weight
        with torch.no_grad():
            w.zero_()
            oc, ic, _, _ = w.shape  # oc == ic == out_ch here
            diag = torch.eye(oc, ic)  # shape [out_ch, out_ch]
            w.copy_(diag.view(oc, ic, 1, 1))

    def forward(self, x):
        identity = self.proj(x)

        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.conv3(y)               

        y = y + identity               
        y = self.post_res_conv(y)      
        y = self.act(y)                
        return y




class DownsampleConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.down = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.down(x)


class UpsampleDeconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.up(x)


class UnetGenerator_3stage(nn.Module):
    def __init__(self, input_nc, output_nc, base_ch=64):
        super().__init__()
        assert base_ch == 64, "This network is fixed to 3 stages with widths 64→128→256."

        ch1 = base_ch          # 64
        ch2 = base_ch * 2      # 128
        ch3 = base_ch * 4      # 256

        # ---------- Encoder ----------
        self.enc1 = ConvBlock3(input_nc, ch1)   # H x W,   64ch
        self.down1 = DownsampleConv(ch1)        # H/2 x W/2

        self.enc2 = ConvBlock3(ch1, ch2)        # H/2 x W/2, 128ch
        self.down2 = DownsampleConv(ch2)        # H/4 x W/4

        self.enc3 = ConvBlock3(ch2, ch3)        # H/4 x W/4, 256ch
        self.down3 = DownsampleConv(ch3)        # H/8 x W/8

        # ---------- Bottleneck (stay at 256ch) ----------
        self.bottleneck = ConvBlock3(ch3, ch3)

        # ---------- Decoder ----------
        self.up3 = UpsampleDeconv(ch3, ch3)     # H/8 -> H/4 (256 -> 256)
        self.dec3 = ConvBlock3(ch3, ch3)        # after skip ADD with enc3

        self.up2 = UpsampleDeconv(ch3, ch2)     # H/4 -> H/2 (256 -> 128)
        self.dec2 = ConvBlock3(ch2, ch2)        # after skip ADD with enc2

        self.up1 = UpsampleDeconv(ch2, ch1)     # H/2 -> H   (128 -> 64)
        self.dec1 = ConvBlock3(ch1, ch1)        # after skip ADD with enc1

        # ---------- Head ----------
        self.out_conv = nn.Conv2d(ch1, output_nc, kernel_size=3, padding=1, bias=True)
        # (No activation here; add e.g. Tanh() externally if needed.)

        # Keep a dedicated ReLU for up-path (mirroring your original style)
        self.up_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)             # 64ch @ H
        d1 = self.down1(e1)

        e2 = self.enc2(d1)            # 128ch @ H/2
        d2 = self.down2(e2)

        e3 = self.enc3(d2)            # 256ch @ H/4
        d3 = self.down3(e3)

        # Bottleneck
        b = self.bottleneck(d3)       # 256ch @ H/8

        # Decoder with additive skips
        u3 = self.up_relu(self.up3(b))     # 256ch @ H/4
        u3 = u3 + e3                       # ADD skip (must match channels)
        u3 = self.dec3(u3)                 # 256ch @ H/4

        u2 = self.up_relu(self.up2(u3))    # 128ch @ H/2
        u2 = u2 + e2
        u2 = self.dec2(u2)                 # 128ch @ H/2

        u1 = self.up_relu(self.up1(u2))    # 64ch @ H
        u1 = u1 + e1
        u1 = self.dec1(u1)                 # 64ch @ H

        out = self.out_conv(u1)            # output_nc @ H
        return out

if __name__ == '__main__':
    model = UnetGenerator(input_nc=3, output_nc=3, num_downs=8)
    print(model)

