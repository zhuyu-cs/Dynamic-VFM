from functools import partial
import torch
import typing as t
from torch import nn, Tensor
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, List, Optional, Tuple
from einops import rearrange


class DropPath(nn.Module):
    """
    Stochastic depth for regularization https://arxiv.org/abs/1603.09382
    Reference:
    - https://github.com/aanna0701/SPT_LSA_ViT/blob/main/utils/drop_path.py
    - https://github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dropout: float = 0.0):
        super(DropPath, self).__init__()
        assert 0 <= dropout <= 1
        self.register_buffer("keep_prop", torch.tensor(1 - dropout))

    def forward(self, inputs: torch.Tensor):
        if self.keep_prop == 1 or not self.training:
            return inputs
        shape = (inputs.size(0),) + (1,) * (inputs.ndim - 1)
        random_tensor = torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
        random_tensor = torch.floor(self.keep_prop + random_tensor)
        outputs = (inputs / self.keep_prop) * random_tensor
        return outputs


class PatchEmbed3d(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (List[int]): Patch token size.
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size: List[int],
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        # padding
        _, _, t, h, w = x.size()
        pad_size = _compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
        x = self.proj(x)  # B C T Wh Ww
        x = x.permute(0, 2, 3, 4, 1)  # B T Wh Ww C
        if self.norm is not None:
            x = self.norm(x)
        return x

class TiedBlockConv3d(nn.Module):
    '''Tied Block Conv3d'''
    def __init__(self, in_planes, planes, kernel_size=7, stride=1, padding=0, bias=True, \
                B=1, groups=1):
        super(TiedBlockConv3d, self).__init__()
        assert planes % B == 0
        assert in_planes % B == 0
        self.B = B
        self.stride = stride
        self.padding = kernel_size//2
        self.out_planes = planes
        self.kernel_size = kernel_size

        self.conv = nn.Conv3d(in_planes//self.B, planes//self.B, 
                    kernel_size=kernel_size, stride=stride, \
                    padding=self.padding, bias=bias, groups=groups)

    def forward(self, x):
        n, c, t, h, w = x.size()
        x = x.reshape(n*self.B, c//self.B, t, h, w)
        x = self.conv(x)
        _, _, t, h, w = x.shape
        x = x.reshape(n, self.out_planes, t, h, w)
        return x

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class MLP(nn.Module):
    def __init__(
        self, 
        in_dim: int,
        hidden_dim: int,
        out_dim: int = None,
        dropout: float = 0.0,
        use_bias: bool = True,
        B=1
    ):
        super().__init__()
        
        if out_dim is None:
            out_dim = in_dim

        self.linear1 = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim*2),
                    SwiGLU(),
                )
        self.dwconv = nn.Sequential(
                    TiedBlockConv3d(hidden_dim, hidden_dim, groups=hidden_dim//B, B=B, kernel_size=3, stride=1),
                    nn.GELU()
                )
        self.linear2 = nn.Sequential(
                    nn.Linear(hidden_dim, out_dim),#, bias=False
                    nn.Dropout(p=dropout)
                )

    def forward(self, x):
        
        #input:  b, t, h, w, c
        x = self.linear1(x)
        x = rearrange(x, ' b t h w c -> b c t h w')
        x = self.dwconv(x)
        x = rearrange(x, ' b c t h w -> b t h w c')
        x = self.linear2(x)

        return x

class BehaviorMLP(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int,
        behavior_t: int,
        dropout: float = 0.0
    ):
        super(BehaviorMLP, self).__init__()
        self.behavior_t = behavior_t
        self.model = self.build_model(
            in_dim=in_dim, out_dim=out_dim, dropout=dropout
        )

    def build_model(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        return nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim // 2, bias=use_bias),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=out_dim // 2, out_features=out_dim // self.behavior_t, bias=use_bias),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor):
        b, t = inputs.shape[0], inputs.shape[1]
        return self.model(inputs).reshape(b, t//self.behavior_t, -1)


def _get_window_and_shift_size(
    shift_size: List[int], size_dhw: List[int], window_size: List[int]
) -> Tuple[List[int], List[int]]:
    for i in range(3):
        if size_dhw[i] <= window_size[i]:
            # In this case, window_size will adapt to the input size, and no need to shift
            window_size[i] = size_dhw[i]
            shift_size[i] = 0

    return window_size, shift_size


torch.fx.wrap("_get_window_and_shift_size")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> Tensor:
    window_vol = window_size[0] * window_size[1] * window_size[2]
    # In 3d case we flatten the relative_position_bias
    relative_position_bias = relative_position_bias_table[
        relative_position_index[:window_vol, :window_vol].flatten()  # type: ignore[index]
    ]
    relative_position_bias = relative_position_bias.view(window_vol, window_vol, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")


def _compute_pad_size_3d(size_dhw: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    pad_size = [(patch_size[i] - size_dhw[i] % patch_size[i]) % patch_size[i] for i in range(3)]
    return pad_size[0], pad_size[1], pad_size[2]


torch.fx.wrap("_compute_pad_size_3d")


def _compute_attention_mask_3d(
    x: Tensor,
    size_dhw: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    shift_size: Tuple[int, int, int],
) -> Tensor:
    # generate attention mask
    attn_mask = x.new_zeros(*size_dhw)
    num_windows = (size_dhw[0] // window_size[0]) * (size_dhw[1] // window_size[1]) * (size_dhw[2] // window_size[2])
    slices = [
        (
            (0, -window_size[i]),
            (-window_size[i], -shift_size[i]),
            (-shift_size[i], None),
        )
        for i in range(3)
    ]
    count = 0
    for d in slices[0]:
        for h in slices[1]:
            for w in slices[2]:
                attn_mask[d[0] : d[1], h[0] : h[1], w[0] : w[1]] = count
                count += 1

    # Partition window on attn_mask
    attn_mask = attn_mask.view(
        size_dhw[0] // window_size[0],
        window_size[0],
        size_dhw[1] // window_size[1],
        window_size[1],
        size_dhw[2] // window_size[2],
        window_size[2],
    )
    attn_mask = attn_mask.permute(0, 2, 4, 1, 3, 5).reshape(
        num_windows, window_size[0] * window_size[1] * window_size[2]
    )
    attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


torch.fx.wrap("_compute_attention_mask_3d")


def shifted_window_attention_3d(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[B, T, H, W, C]): The input tensor, 5-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): 3-dimensions window size, T, H, W .
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention (T, H, W).
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[B, T, H, W, C]: The output tensor after shifted window attention.
    """
    b, t, h, w, c = input.shape
    # pad feature maps to multiples of window size
    pad_size = _compute_pad_size_3d((t, h, w), (window_size[0], window_size[1], window_size[2]))
    x = F.pad(input, (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
    _, tp, hp, wp, _ = x.shape
    padded_size = (tp, hp, wp)

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    # partition windows
    num_windows = (
        (padded_size[0] // window_size[0]) * (padded_size[1] // window_size[1]) * (padded_size[2] // window_size[2])
    )
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        window_size[0],
        padded_size[1] // window_size[1],
        window_size[1],
        padded_size[2] // window_size[2],
        window_size[2],
        c,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        b * num_windows, window_size[0] * window_size[1] * window_size[2], c
    )  # B*nW, Wd*Wh*Ww, C

    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (c // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask to handle shifted windows with varying size
        attn_mask = _compute_attention_mask_3d(
            x,
            (padded_size[0], padded_size[1], padded_size[2]),
            (window_size[0], window_size[1], window_size[2]),
            (shift_size[0], shift_size[1], shift_size[2]),
        )
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        padded_size[1] // window_size[1],
        padded_size[2] // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        c,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, tp, hp, wp, c)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

    # unpad features
    x = x[:, :t, :h, :w, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention_3d")


class ShiftedWindowAttention3d(nn.Module):
    """
    See :func:`shifted_window_attention_3d`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(window_size) != 3 or len(shift_size) != 3:
            raise ValueError("window_size and shift_size must be of length 2")

        self.window_size = window_size  # Wd, Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self) -> None:
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                self.num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self) -> None:
        # get pair-wise relative position index for each token inside the window
        coords_dhw = [torch.arange(self.window_size[i]) for i in range(3)]
        coords = torch.stack(
            torch.meshgrid(coords_dhw[0], coords_dhw[1], coords_dhw[2], indexing="ij")
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        # We don't flatten the relative_position_index here in 3d case.
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self, window_size: List[int]) -> torch.Tensor:
        return _get_relative_position_bias(self.relative_position_bias_table, self.relative_position_index, window_size)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        _, t, h, w, _ = x.shape
        size_dhw = [t, h, w]
        window_size, shift_size = self.window_size.copy(), self.shift_size.copy()
        # Handle case where window_size is larger than the input tensor
        window_size, shift_size = _get_window_and_shift_size(shift_size, size_dhw, window_size)

        relative_position_bias = self.get_relative_position_bias(window_size)

        return shifted_window_attention_3d(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            window_size,
            self.num_heads,
            shift_size=shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )

class Transformer(nn.Module):
    def __init__(
        self,
        window_size: t.Tuple[int, int, int],
        behavior_t: int,
        behavior_dim: int,
        emb_dim: int,
        num_blocks: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float =0.0,
        drop_path: float = 0.0,
        use_bias: bool = True,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.grad_checkpointing = grad_checkpointing
        for i in range(num_blocks):
            block = nn.ModuleDict(
                {
                    "mha": ShiftedWindowAttention3d(
                        dim=emb_dim, 
                        attention_dropout=attention_dropout,                       
                        dropout=dropout,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=[w // 2 for w in window_size],
                    ),
                    "mlp": MLP(
                        in_dim=emb_dim,
                        hidden_dim=mlp_dim,
                        dropout=dropout,
                        use_bias=use_bias,
                    )
                }
            )
            block["b-mlp"] = BehaviorMLP(
                in_dim=behavior_dim,
                behavior_t = behavior_t,
                out_dim=emb_dim
            )
            self.blocks.append(block)
        self.drop_path = DropPath(dropout=drop_path)

        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def checkpointing(self, fn: t.Callable, inputs: torch.Tensor):
        if self.grad_checkpointing:
            outputs = checkpoint(
                fn, inputs, preserve_rng_state=True, use_reentrant=False
            )
        else:
            outputs = fn(inputs)
        return self.drop_path(outputs) + inputs

    def forward(
        self,
        inputs: torch.Tensor,
        behaviors: torch.Tensor,
    ):
        output = inputs
        for block in self.blocks:
            if "b-mlp" in block:
                b_latent = block["b-mlp"](behaviors).unsqueeze(-2).unsqueeze(-2)
                output = output + b_latent
            
            output = self.checkpointing(block["mha"], output)#block["mha"](output)#
            output = self.drop_path(block["mlp"](output)) + output
        return output


class SwinCore(nn.Module):
    
    def __init__(
        self,
        in_channels=1,
        behavior_dim=4,
        patch_size=[2, 4, 4],
        window_size=[8, 7, 7],
        emb_dim=160,
        num_blocks=4,
        num_heads=4,
        mlp_dim=488,
        t_dropout=0.0, 
        drop_path=0.0,
        bias=True,
        grad_checkpointing=False,
    ):
        super(SwinCore, self).__init__()
        
        self.patch_embedding = PatchEmbed3d(
            in_channels=in_channels,
            patch_size=patch_size, 
            embed_dim=emb_dim, 
            norm_layer=partial(nn.LayerNorm, eps=1e-5))

        self.transformer = Transformer(
            behavior_dim=behavior_dim,
            behavior_t=patch_size[0],
            window_size=window_size,
            emb_dim=emb_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=t_dropout,
            drop_path=drop_path,
            use_bias=bias,
            grad_checkpointing=grad_checkpointing,
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):  
        # inputs: B C T H W
        output = self.patch_embedding(inputs) # B _T _H _W C
        behaviors = torch.cat((behaviors, pupil_centers), dim=1) # B 4 T 

        output = self.transformer(output, behaviors=behaviors.transpose(1,2)) # B _T _H _W C
        
        return output.permute(0, 4, 1, 2, 3)  # B, C, _T, _H, _W

    