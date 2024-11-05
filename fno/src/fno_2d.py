import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralOperator2d(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 out_channels=1,
                 modes=[50, 100]):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.weights_pos = nn.Parameter(torch.empty((in_channels, out_channels, *modes), dtype=torch.cfloat))
        self.weights_neg = nn.Parameter(torch.empty((in_channels, out_channels, *modes), dtype=torch.cfloat))

        nn.init.xavier_uniform_(self.weights_pos)
        nn.init.xavier_uniform_(self.weights_neg)

    def multiply_weights(self, x, weights):
        return torch.einsum('bihw,iohw->bohw', x, weights)
    
    def forward(self, x, pad=0):
        '''
        Padding may be included to handle non-periodic domains
        '''
        b, c, h, w = x.shape

        # Apply Fourier transform
        pad_size = [h + pad, w + pad]
        x_q = torch.fft.rfft2(x, s=pad_size)

        out_size = [b, self.out_channels, pad_size[0], pad_size[1]//2 + 1]

        # Apply spectral weights
        x_out = torch.zeros(out_size, dtype=torch.cfloat, device=x.device)
        x_out[:, :, :self.modes[0], :self.modes[1]] = \
            self.multiply_weights(x_q[:, :, :self.modes[0], :self.modes[1]], self.weights_pos)
        x_out[:, :, -self.modes[0]:, :self.modes[1]] = \
            self.multiply_weights(x_q[:, :, -self.modes[0]:, :self.modes[1]], self.weights_neg)

        # Inverse Fourier transform
        x = torch.fft.irfft2(x_out, s=pad_size)

        # Remove any zero-padding
        if pad > 0:
            x = x[:, :, :-pad, :-pad]

        return x

class FNOBlock2d(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 modes=[50, 100],
                 bn=True,
                 act=nn.ReLU()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bn = bn

        self.spectral_conv = SpectralOperator2d(in_channels, out_channels, modes)
        #FNO paper uses a conv layer with kernel size = 1, which is the same as
        # a linear layer on the channel dimension
        self.local_func = nn.Linear(in_channels, out_channels)
        if bn:
            self.bn = nn.BatchNorm2d(self.out_channels)
        else:
            self.bn = nn.Identity()
        self.act=act

    def forward(self, x, pad=0):
        b, c, h, w = x.shape

        x1 = self.spectral_conv(x, pad=pad) #[b, c, h, w]
        x2 = self.local_func(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x = self.bn(x1 + x2)
        x = self.act(x)

        return x

class FNO2d(nn.Module):
    '''
    Deep nonlinear Fourier neural operator
    Solves 2D PDEs
    '''
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 fno_channels=32,
                 modes=[16, 16],
                 n_fno_layers=4,
                 fno_pad=0,
                 fno_act=nn.ReLU()):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fno_channels = fno_channels
        self.modes = modes
        self.n_fno_layers = n_fno_layers
        self.fno_pad = fno_pad
        self.fno_act = fno_act

        fno_blocks = [
            FNOBlock2d(self.fno_channels, self.fno_channels, modes=self.modes, act=self.fno_act) \
            for i in range(self.n_fno_layers)
        ]
        self.fno = nn.ModuleList(fno_blocks)

        self.read_in = nn.Linear(self.in_channels, self.fno_channels)
        self.read_out = nn.Linear(self.fno_channels, self.out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels

        x = x.permute(0, 2, 3, 1)
        x = self.read_in(x)

        x = x.permute(0, 3, 1, 2)
        for fno_block in self.fno:
            x = fno_block(x, pad=self.fno_pad)

        x = x.permute(0, 2, 3, 1)
        x = self.read_out(x)
        x = x.permute(0, 3, 1, 2)

        return x
