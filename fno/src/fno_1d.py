import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralOperator1d(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 modes=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.weights = nn.Parameter(torch.empty((in_channels, out_channels, modes), dtype=torch.cfloat))
        nn.init.xavier_uniform_(self.weights)
    
    def multiply_weights(self, x, weights):
        return torch.einsum('bil,iol->bol', x, weights)

    def forward(self, x, pad=0):
        '''
        Padding may be included to handle non-periodic domains
        '''
        b, c, l = x.shape

        pad_size = l+pad
        x_q = torch.fft.rfft(x, n=pad_size)

        out_size = [b, self.out_channels, pad_size//2+1]
        x_out = torch.zeros(out_size, dtype=torch.cfloat, device=x.device)
        x_out[:, :, :self.modes] = self.multiply_weights(x_q[:, :, :self.modes], self.weights)

        x = torch.fft.irfft(x_out, n=pad_size)

        # Remove any zero-padding
        if pad > 0:
            x = x[:, :, :-pad]

        return x

class FNOBlock1d(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 modes=16,
                 act=nn.ReLU()):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.spectral_conv = SpectralOperator1d(in_channels, out_channels, modes)
        #FNO paper uses a conv layer with kernel size = 1, which is the same as
        # a linear layer on the channel dimension
        self.local_func = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.act = act

    def forward(self, x, pad=0):
        b, c, l = x.shape

        x1 = self.spectral_conv(x, pad=pad) #[b, c, l]
        x2 = self.local_func(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.bn(x1 + x2)
        x = self.act(x)

        return x
    
class FNO1d(nn.Module):
    '''
    Deep nonlinear Fourier neural operator
    Solves 1D PDEs
    '''
    def __init__(self,
                 in_channels=2,
                 out_channels=1,
                 fno_channels=32,
                 modes=16,
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
            FNOBlock1d(self.fno_channels, self.fno_channels, modes=self.modes, act=self.fno_act) \
            for i in range(self.n_fno_layers)
        ]
        self.fno = nn.ModuleList(fno_blocks)

        self.read_in = nn.Linear(self.in_channels, self.fno_channels)
        self.read_out = nn.Linear(self.fno_channels, self.out_channels)

    def forward(self, x):
        b, c, l = x.shape
        assert c == self.in_channels

        x = x.permute(0, 2, 1)
        x = self.read_in(x)

        x = x.permute(0, 2, 1)
        for fno_block in self.fno:
            x = fno_block(x, pad=self.fno_pad)

        x = x.permute(0, 2, 1)
        x = self.read_out(x)
        x = x.permute(0, 2, 1)

        return x