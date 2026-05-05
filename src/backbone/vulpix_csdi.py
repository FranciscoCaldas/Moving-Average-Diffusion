import math
import torch
import torch.nn as nn
from src.utils.vulpix_utils import calc_diffusion_step_embedding


def swish(x):
    return x * torch.sigmoid(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out
    
def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu")
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class FeatureTransformer(nn.Module):
    def __init__(self, seq_len, heads=8, ff_dim=128):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=seq_len,         # Treat seq_len as the token length
            nhead=heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu"         # Input shape: (B, D, T)
        )

    def forward(self, x):
        # x: (B, D, T) where D is the feature dimension and T is the sequence length
        #x = x.transpose(1, 2)         # Now shape (B, D, T)
        out = self.transformer(x)     # Self-attention across features
        #out = out.transpose(1, 2)     # Back to (B, T, D)
        return out
    
class TimeTransformer(nn.Module):
    def __init__(self, channels, heads=8, ff_dim=128):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(
            d_model=channels,         # Treat seq_len as the token length
            nhead=heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="gelu"         # Input shape: (B, D, T)
        )

    def forward(self, x):
        # x: (B, D, T)
        x = x.transpose(1, 2)         # Now shape (B, D, T)
        out = self.transformer(x)     # Self-attention across features
        out = out.transpose(1, 2)     # Back to (B, T, D)
        return out

    
class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels,  
                 diffusion_step_embed_dim_out, in_channels,seq_len,ff_dim):
        super(Residual_block, self).__init__()
        
        self.res_channels = res_channels
        # the layer-specific fc for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        
        # dilated conv layer
        #self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)
        #find head value
        #self.heads = int(math.sqrt(self.seq_len))
        h = 0
        self.heads = 8
        while h == 0:
            if seq_len % (self.heads) == 0:
                h = self.heads
            else:
                self.heads += 1
        
        self.time_trans = TimeTransformer(heads=8, channels=self.res_channels,ff_dim=ff_dim)
        self.feat_trans = FeatureTransformer(heads=h, seq_len=seq_len,ff_dim = ff_dim)


        
        # add mel spectrogram upsampler and conditioner conv1x1 layer  (In adapted to S4 output)
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)
        self.double_conv = Conv(self.res_channels,2*self.res_channels,kernel_size=1)  # 80 is mel bands

        # residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

        
    def forward(self, input_data):
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        #assert C == self.res_channels
        #assert L == self.                      
                                                           
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])    
        h = h + part_t
        
        h = self.time_trans(h)
        h = self.feat_trans(h)

        h = self.double_conv(h)  # (B, 2*res_channels, L)
        #h = self.dilated_conv_layer(h)
        # add (local) conditioner
        assert cond is not None

        cond = self.cond_conv(cond)
        h += cond  
        
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])   

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  

    
    

class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,seq_len,ff_dim):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        
        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)

      
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,seq_len=seq_len,ff_dim=ff_dim))

    def forward(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed)) 
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)  # normalize for training stability


class CSDIImputer(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,seq_len,ff_dim):
        super(CSDIImputer, self).__init__()

        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())

        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             ff_dim = ff_dim,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             seq_len=seq_len)
        
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data):

        noise, conditional, mask, diffusion_steps = input_data 
        
       
        conditional = conditional * mask
        conditional = torch.cat([conditional, mask.float()], dim=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y

