import torch
import torch.nn as nn
import torch.nn.functional as F

def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.)

class ConvNextBlock(nn.Module):
    ''' Convolutional block, ignoring layernorm for now 
    '''
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout_rate=0.1):
        super().__init__()

        self.depth_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            padding='same', padding_mode='replicate', groups=in_channels)
        self.conv1 = nn.Conv2d(out_channels, 4*out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(4*out_channels, out_channels, kernel_size=1)
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = torch.sin(x)
        x = self.conv2(x)
        
        x = self.dropout(x)
        return x

class LatentNet(nn.Module):
    ''' Neural network architecture used in Cell PBNN work
    '''
    def __init__(self,
                input_dims=3,
                output_dims=1,
                hidden_dims=64,
                num_hidden=8,
                dropout_rate=0.1):
        
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate

        self.read_in = nn.Sequential(
            nn.Conv2d(input_dims, 4, kernel_size=1),
            ConvNextBlock(4, hidden_dims, dropout_rate=dropout_rate)
        )
        self.downsample = nn.Sequential(
                nn.Conv2d(hidden_dims, hidden_dims, kernel_size=4, stride=4),
                nn.GELU()
            )
        self.read_out = nn.Conv2d(2*hidden_dims, output_dims, kernel_size=1)

        self.cnn1 = nn.ModuleList()
        self.cnn2 = nn.ModuleList()
        for i in range(num_hidden):
            self.cnn1.append(ConvNextBlock(hidden_dims, hidden_dims, dropout_rate=dropout_rate))
            self.cnn2.append(ConvNextBlock(hidden_dims, hidden_dims, dropout_rate=dropout_rate))
    
    def forward(self, x):
        x = self.read_in(x)
        for cell in self.cnn1:
            x = x + cell(x)
        
        latent = self.downsample(x)
        for cell in self.cnn2:
            latent = latent + cell(latent)
        latent = F.interpolate(latent, x.shape[-2:])

        x = torch.cat([x, latent], dim=1)
        x = self.read_out(x)

        return x

class UNetEncoder(nn.Module):
    """ UNet encoder - preserves outputs and sends across latent bottleneck
    """
    def __init__(self,
                input_dims=3,
                stage_dims=8,
                blocks_per_stage=2,
                num_stages=3,
                conv_stride=2):
        super().__init__()
        self.downsample_blocks = nn.ModuleList()
        self.stages = nn.ModuleList()

        self.read_in = nn.Sequential(
            nn.Conv2d(input_dims, stage_dims, kernel_size=3, padding='same'),
            *[ConvNextBlock(stage_dims, stage_dims) for j in range(blocks_per_stage)])

        for i in range(num_stages):
            stage = nn.Sequential(
                nn.BatchNorm2d(stage_dims),
                nn.Conv2d(stage_dims, stage_dims*2, kernel_size=conv_stride, stride=conv_stride)
            )
            self.downsample_blocks.append(stage)
            stage_dims = stage_dims * 2
            
            stage = nn.ModuleList([ConvNextBlock(stage_dims, stage_dims) for j in range(blocks_per_stage)])
            self.stages.append(stage)

    def forward(self, x):
        encoder_outputs = []
        x = self.read_in(x)
        for i in range(len(self.stages)):
            encoder_outputs.append(x)
            x = self.downsample_blocks[i](x)
            for cell in self.stages[i]:
                x = x + cell(x)
        return x, encoder_outputs

class UNetDecoder(nn.Module):
    """ UNet Decoder - accepts incoming skip connections
    """
    def __init__(self,
                output_dims, 
                stage_dims=8,
                blocks_per_stage=2,
                num_stages=3,
                conv_stride=2):
        super().__init__()
        self.read_out = nn.Conv2d(stage_dims, output_dims, kernel_size=1)

        self.upsample_blocks = nn.ModuleList()
        self.combiners = nn.ModuleList()
        self.stages = nn.ModuleList()

        stage_dims = stage_dims * (2**num_stages)

        for i in range(num_stages):
            stage = nn.Sequential(
                nn.BatchNorm2d(stage_dims),
                nn.ConvTranspose2d(stage_dims, stage_dims//2, kernel_size=conv_stride, stride=conv_stride)
            )
            self.upsample_blocks.append(stage)
            stage_dims = stage_dims // 2

            self.combiners.append(nn.Conv2d(2*stage_dims, stage_dims, kernel_size=1))

            stage = nn.ModuleList([ConvNextBlock(stage_dims, stage_dims) for j in range(blocks_per_stage)])
            self.stages.append(stage)
    
    def forward(self, x, encoder_outputs):
        for i in range(len(self.stages)):
            x = self.upsample_blocks[i](x)
            x2 = encoder_outputs[-(i+1)]
            diffY = x2.size()[-2] - x.size()[-2]
            diffX = x2.size()[-1] - x.size()[-1]
            x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                            diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, x2], dim=-3)
            x = self.combiners[i](x)
            for cell in self.stages[i]:
                x = x + cell(x)
        x = self.read_out(x)
        return x

class UNet(torch.nn.Module):
    """ UNet - encoder decoder architecture with skip connections
        across latent bottleneck.
    """
    def __init__(self,
                input_dims=3,
                output_dims=1,
                stage_dims=16,
                blocks_per_stage=2,
                num_stages=4,
                conv_stride=4):
        super().__init__()
        self.encoder = UNetEncoder(input_dims, stage_dims, blocks_per_stage, num_stages, conv_stride)
        self.decoder = UNetDecoder(output_dims, stage_dims, blocks_per_stage, num_stages, conv_stride)
        self.apply(_init_weights)

    def forward(self, x):
        _, _, h0, w0 = x.shape
        x, en_outputs = self.encoder(x) #[B, C, H, W]
        y = self.decoder(x, en_outputs)
        
        return y