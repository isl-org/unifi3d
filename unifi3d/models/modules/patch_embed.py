from itertools import repeat
import numpy as np
import torch
import torch.nn as nn
import omegaconf

try:
    from torch import _assert
except ImportError:

    def _assert(condition: bool, message: str):
        assert condition, message


from unifi3d.utils.utils import _ntuple

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: tuple of (height, width)
    return:
    pos_embed: [height*width, embed_dim] or [1+height*width, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid [height, width, depth]
    return:
    pos_embed: [height*width*depth, embed_dim] or [1+height*width*depth, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid_d = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, grid_d)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    # assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])
    emb_d = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])

    emb = np.concatenate([emb_h, emb_w, emb_d], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Tokenizer(nn.Module):
    """Abstract tokenizer class"""

    def __init__(self):
        self.patch_volume = 0  # required for output layer in DiT

    def forward(self, x):
        pass

    def unpatchify(self, x, out_channels=1):
        pass

    def get_pos_embed(self, hidden_size):
        """Get positional embedding of size (1, num_patches, hidden_size)"""
        pass


class DummyTokenizer(nn.Module):
    """
    Dummy tokenizer for representations that are already in token format
    e.g., 3DShape2VecSet encoding already provides a set of tokens, no need to patchify
    """

    def __init__(self, n_tokens: int, hidden_size: int, reshape_flat_input=False):
        """
        Args:
            n_tokens: Number of tokens
            hidden_size: Hidden size of each token (embedding dimension)
            reshape_flat_input: If True, the input is assumed to be flat and is first
                reshaped to the correctd dimensions (n_tokens, hidden_size)
        """
        super().__init__()
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size
        self.patch_volume = hidden_size
        self.reshape_flat_input = reshape_flat_input

    def forward(self, x):
        if self.reshape_flat_input:
            x = x.reshape(-1, self.n_tokens, self.hidden_size)
        return x

    def unpatchify(self, x, out_channels=1):
        x = x.reshape(-1, self.n_tokens, self.hidden_size, out_channels)  # N, T, D, C
        x = torch.einsum("ntdc->nctd", x)
        if self.reshape_flat_input:
            x = x.reshape(-1, out_channels, self.n_tokens * self.hidden_size)
        # if only one channel, squeeze it out
        return torch.squeeze(x, 1)

    def get_pos_embed(self, hidden_size):
        """Get positional embedding of size (1, num_patches, hidden_size)"""
        # dummy tokenizer does not require positional embedding -> return zeros
        pos_embed = torch.zeros(1, self.n_tokens, hidden_size)
        return pos_embed


class PatchEmbed3D(nn.Module):
    """
    3D to Patch Embedding
    Apply 3D conv to embed input into patches
    """

    def __init__(
        self,
        img_size=16,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = (img_size, img_size, img_size)
        patch_size = (patch_size, patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.flatten = flatten
        self.add_pos_embed = True

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        _assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    def unpatchify(self, x, out_channels=1):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = out_channels
        p = self.patch_size[0]
        h = w = d = int(round(x.shape[1] ** (1 / 3)))
        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, c))
        x = torch.einsum("nhwdpqrc->nchpwqdr", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p, h * p))
        return imgs

    def get_pos_embed(self, hidden_size):
        """Get positional embedding of size (1, num_patches, hidden_size)"""
        # initialize parameter
        pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size),
            requires_grad=False,
        )
        pos_embed_sin_cos = get_3d_sincos_pos_embed(hidden_size, self.grid_size)
        pos_embed.data.copy_(torch.from_numpy(pos_embed_sin_cos).float().unsqueeze(0))
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)
        return pos_embed


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, list) or isinstance(
            img_size, omegaconf.listconfig.ListConfig
        ):
            img_size = tuple(img_size)
        else:
            raise ValueError(
                f"img_size must be int or list/tuple, but got {type(img_size)}"
            )

        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        assert (
            self.img_size[0] % patch_size[0] == 0
        ), "1st image dimensions must be divisible by the patch size."
        assert (
            self.img_size[1] % patch_size[1] == 0
        ), "2nd image dimensions must be divisible by the patch size."

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_volume = patch_size[0] * patch_size[1]
        self.flatten = flatten
        self.add_pos_embed = True

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape  # A way to throw asserts
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        # _assert(
        #     W == self.img_size[1],
        #     f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        # )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

    def unpatchify(self, x, out_channels=1):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = out_channels
        p = self.patch_size[0]
        h = self.grid_size[0]
        w = self.grid_size[1]
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def get_pos_embed(self, hidden_size):
        """Get positional embedding of size (1, num_patches, hidden_size)"""
        # initialize parameter
        pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size),
            requires_grad=False,
        )

        pos_embed_sin_cos = get_2d_sincos_pos_embed(hidden_size, self.grid_size)
        pos_embed.data.copy_(torch.from_numpy(pos_embed_sin_cos).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj.bias, 0)

        return pos_embed
