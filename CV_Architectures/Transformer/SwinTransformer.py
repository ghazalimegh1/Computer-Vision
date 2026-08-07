import torch
from torch import nn

class SwinTransformer(nn.Module):
    def __init__(self, layers, d_model, num_heads, patch_size,window_size, shift_size):
        super().__init__()
        self.layers = layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.architecture = nn.ModuleList(nn.ModuleList(SwinTransformerLayer(d_model,
                                                  num_heads, 
                                                  patch_size,
                                                  window_size, 
                                                  shift_size, 
                                                  patch_embedding=True if self.layers.index(stage)==0 else False) for layer in stage)
                                                  for stage in self.layers)


    def forward(self, x):
        features_map = []
        for stage in self.architecture:
            for layer in stage:
                x = layer(x)
            features_map.append(x)

        return x, features_map


class SwinTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, patch_size,window_size, shift_size, patch_embedding=False):
        super().__init__()
        self.patch_merging = PatchMerging(d_model, d_model * 2)
        self.swin_transformer_block = SwinTransformerBlock(d_model, num_heads, window_size, shift_size)
        self.patch_embedding = patch_embedding
        self.patch_embedder = PatchEmbedding(in_channels=3, embed_dim=d_model, patch_size=patch_size)


    def forward(self, x):
        if self.patch_embedding:
            x = self.patch_embedder(x)
        else:
            x = self.patch_merging(x)
        x = self.swin_transformer_block(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, window_size):
        super().__init__()
        self.w_msa = WindowAttention(d_model, num_heads, window_size)
        self.sw_msa = WindowAttention(d_model, num_heads, window_size, shifted=True)

    def forward(self, x):
        x = self.w_msa.forward(x)
        x = self.sw_msa.forward(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels * 4, out_channels)
        

    def forward(self, x):
        batch, num_patches, c = x.size()
        H = W = int(num_patches ** 0.5)
        x = x.view(batch, H, W, c)
        x = x.view(batch, H // 2, 2, W // 2, 2, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch, H // 2, W // 2, 4 * c)
        x = x.view(batch, -1, 4 * c)
        x = self.linear(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embed_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size)
        self.flatten = nn.Flatten(2)

    def forward(self, x):
        x = self.patcher(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        return x
        

class WindowAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size, shifted=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        self.window_partition = nn.Unfold(kernel_size=window_size, stride=window_size)
        self.shift_size = window_size // 2
        self.shifted = shifted
        self.query_linear = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(self.num_heads)])
        self.key_linear = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(self.num_heads)])
        self.value_linear = nn.ModuleList([nn.Linear(d_model, self.head_dim) for _ in range(self.num_heads)])
        self.relative_position_bias = nn.Parameter(torch.zeros(1, num_heads, window_size * window_size, window_size * window_size))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )


    def forward_no_shifted(self, x):
        batch_size, num_patches, c = x.size()
        x = self.norm1(x)
        x = x.view(batch_size, int(num_patches ** 0.5) , int(num_patches ** 0.5), c)
        windows = self.window_partition(x.permute(0, 3, 1, 2))
        windows = windows.view(batch_size, -1, self.window_size * self.window_size, c)
        query = [self.query_linear[i](windows) for i in range(self.num_heads)]
        key = [self.key_linear[i](windows) for i in range(self.num_heads)]
        value = [self.value_linear[i](windows) for i in range(self.num_heads)]

        head_outputs = []
        for i in range(self.num_heads):
            attention_scores = torch.matmul(query[i], key[i].transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention_scores = attention_scores + self.relative_position_bias[:, i:i + 1, :, :]
            attention_weights = nn.functional.softmax(attention_scores, dim=-1)
            head_output = torch.matmul(attention_weights, value[i])
            head_outputs.append(head_output)

        attention_output = torch.cat(head_outputs, dim=-1)
        attention_output = attention_output.view(batch_size, -1, self.window_size * self.window_size, c)
         # size of attention_output: (batch_size, num_windows, window_size * window_size, d_model)
        H = W = int(num_patches ** 0.5)
        num_windows_per_side = H // self.window_size

        attention_output = attention_output.view(
            batch_size,
            num_windows_per_side,
            num_windows_per_side,
            self.window_size,
            self.window_size,
            c
        )

        attention_output = attention_output.permute(
            0, 1, 3, 2, 4, 5
        )

        attention_output = attention_output.reshape(
            batch_size,
            H,
            W,
            c
        )
        attention_output = attention_output.view(batch_size, num_patches, c)
        #size of attention_output: (batch_size, num_patches, d_model)
        

        windows = x.view(batch_size, num_patches, c)
        z_hat = windows + attention_output
        temp = self.norm2(z_hat)
        temp = self.mlp(temp)
        z = z_hat + temp
        z = z.view(batch_size, -1, c) 
        return z

    
    def forward_shifted(self, x):
        batch_size, num_patches, c = x.size()
        x = self.norm1(x)
        x = x.view(batch_size, int(num_patches ** 0.5) , int(num_patches ** 0.5), c)
        x = torch.roll(
                        x,
                        shifts=(-self.shift_size,-self.shift_size),
                        dims=(1,2)
                    )
        windows = self.window_partition(x)
        windows = windows.view(batch_size, -1, self.window_size * self.window_size, c)
        query = [self.query_linear[i](windows) for i in range(self.num_heads)]
        key = [self.key_linear[i](windows) for i in range(self.num_heads)]
        value = [self.value_linear[i](windows) for i in range(self.num_heads)]
        head_outputs = []
        for i in range(self.num_heads):
            attention_scores = torch.matmul(query[i], key[i].transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention_scores = attention_scores + self.relative_position_bias[:, i:i + 1, :, :]
            attention_weights = nn.functional.softmax(attention_scores + self.mask, dim=-1)
            head_output = torch.matmul(attention_weights, value[i])
            head_outputs.append(head_output)
        attention_output = torch.cat(head_outputs, dim=-1)
        attention_output = attention_output.view(batch_size, -1, self.window_size * self.window_size, c)
         # size of attention_output: (batch_size, num_windows, window_size * window_size, d_model)
        H = W = int(num_patches ** 0.5)
        num_windows_per_side = H // self.window_size
        attention_output = attention_output.view(
            batch_size,
            num_windows_per_side,
            num_windows_per_side,
            self.window_size,
            self.window_size,
            c
        )
        attention_output = attention_output.permute(
            0, 1, 3, 2, 4, 5
        )
        attention_output = attention_output.reshape(
            batch_size,
            H,
            W,
            c
        )
        attention_output = attention_output.view(batch_size, num_patches, c)
        #size of attention_output: (batch_size, num_patches, d_model)

        windows = x.view(batch_size, num_patches, c)
        z_hat = windows + attention_output
        temp = self.norm2(z_hat)
        temp = self.mlp(temp)
        z = z_hat + temp
        z = z.view(batch_size, -1, c) 
        return z

    
    def create_mask(self, H, M, S):
        # create 2D Grid
        img_mask = torch.zeros((1, H, H, 1), dtype=torch.int64)

        h_slices = (
            slice(0, -M),
            slice(-M, -S),
            slice(-S, None)
        )

        w_slices = (
            slice(0, -M),
            slice(-M, -S),
            slice(-S, None)
        )

        cnt = 0

        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # shift 2D Grid

        img_mask = torch.roll(img_mask, (-S, -S), dims=(1, 2))


        # partition the result into windows
        img_mask_windows = self.window_partition(img_mask.permute(0,3,1,2))
        img_mask_windows = img_mask_windows.view(-1, M * M)
        img_mask_windows = (
                           img_mask_windows.unsqueeze(1)
                           -
                           img_mask_windows.unsqueeze(2)
                            )

        img_mask_windows = img_mask_windows.masked_fill(
            img_mask_windows != 0, 
            float(-100.0)
        )

        img_mask_windows = img_mask_windows.masked_fill(
                    img_mask_windows == 0, 
                    float(0.0)
                )

        return img_mask_windows


        # for each window i create the matrix of masks
    

    def forward(self, x):
        if self.shifted:
            return self.forward_shifted(x)
        return self.forward_no_shifted(x)


    


