# models.py - the three models, all with the same 80/20 two-branch design
import torch
import torch.nn as nn


# ---------- building blocks ----------
class ConvBlock(nn.Module):
    """Two 3x3 convolutions (the basic unit of a U-Net)."""
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net: encoder (shrinks the map) -> decoder (grows it back) with skip links.
    Input height/width must be divisible by 4."""
    def __init__(self, cin, cout, base=16):
        super().__init__()
        self.e1 = ConvBlock(cin, base)
        self.e2 = ConvBlock(base, base * 2)
        self.bott = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, cout, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(b), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


class ConvLSTM(nn.Module):
    """An LSTM whose gates are convolutions, so it reads whole maps one year at a time."""
    def __init__(self, cin, hid, k=3):
        super().__init__()
        self.hid = hid
        self.gates = nn.Conv2d(cin + hid, 4 * hid, k, padding=k // 2)

    def forward(self, seq):                          # seq: (B, T, C, H, W)
        B, T, _, H, W = seq.shape
        h = seq.new_zeros(B, self.hid, H, W)         # memory starts empty
        c = seq.new_zeros(B, self.hid, H, W)
        for t in range(T):                           # 1993 -> 2003 -> 2013
            i, f, o, g = torch.chunk(self.gates(torch.cat([seq[:, t], h], 1)), 4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
        return h                                     # summary of the whole history


# ---------- land-cover branches (one per model) ----------
class UNetLC(nn.Module):
    def __init__(self, n_classes, n_steps, feat):
        super().__init__()
        self.net = UNet(n_steps * n_classes, feat)

    def forward(self, seq):                          # years stacked as channels
        return self.net(seq.flatten(1, 2))


class PixelLSTMLC(nn.Module):
    def __init__(self, n_classes, n_steps, feat):
        super().__init__()
        self.lstm = nn.LSTM(n_classes, feat, batch_first=True)

    def forward(self, seq):                          # every pixel = its own sequence
        B, T, n, H, W = seq.shape
        x = seq.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, n)
        out, _ = self.lstm(x)
        return out[:, -1].reshape(B, H, W, -1).permute(0, 3, 1, 2)


class ConvLSTMUNetLC(nn.Module):
    def __init__(self, n_classes, n_steps, feat, hid=16):
        super().__init__()
        self.clstm = ConvLSTM(n_classes, hid)
        self.unet = UNet(hid, feat)

    def forward(self, seq):
        return self.unet(self.clstm(seq))


# ---------- auxiliary branch (slope, later elevation) ----------
class AuxBranch(nn.Module):
    def __init__(self, n_aux, feat, k):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(n_aux, 16, k, padding=k // 2),
                                 nn.ReLU(inplace=True), nn.Conv2d(16, feat, 1))

    def forward(self, x):
        return self.net(x)


# ---------- the full model with the fixed 80/20 fusion ----------
class Forecaster(nn.Module):
    def __init__(self, kind, n_classes, n_steps, n_aux, feat=32, lc_weight=0.8):
        super().__init__()
        backbones = {"unet": UNetLC, "pixel_lstm": PixelLSTMLC, "convlstm_unet": ConvLSTMUNetLC}
        self.lc = backbones[kind](n_classes, n_steps, feat)
        # the per-pixel LSTM must not see neighbours -> its aux branch uses 1x1 convs
        self.aux = AuxBranch(n_aux, feat, k=1 if kind == "pixel_lstm" else 3)
        # normalisation WITHOUT learnable scale, so 0.8/0.2 really limits the influence
        self.norm_lc = nn.BatchNorm2d(feat, affine=False)
        self.norm_aux = nn.BatchNorm2d(feat, affine=False)
        self.w_lc, self.w_aux = lc_weight, 1.0 - lc_weight
        self.head = nn.Conv2d(feat, n_classes, 1)

    def forward(self, seq, aux):                     # seq (B,T,n,H,W), aux (B,C,H,W)
        f_lc = self.norm_lc(self.lc(seq))
        f_aux = self.norm_aux(self.aux(aux))
        return self.head(self.w_lc * f_lc + self.w_aux * f_aux)