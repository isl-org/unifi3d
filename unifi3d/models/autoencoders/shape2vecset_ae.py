# --------------------------------------------------------
# Code adapted from https://github.com/1zb/3DShape2VecSet/
# Licensed under The MIT License
# --------------------------------------------------------

from einops import rearrange, repeat
from functools import wraps
import numpy as np
from timm.models.layers import DropPath
import torch
from torch_cluster import fps

from unifi3d.models.autoencoders.base_encoder_decoder import BaseEncoderDecoder


def init_weights(net, init_type="normal", gain=0.01) -> None:
    def init_func(m) -> None:
        classname = m.__class__.__name__
        if classname.find("BatchNorm2d") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "none":  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # Propagate to children
    for m in net.children():
        m.apply(init_func)


def exists(val):
    """
    Check if a value exists (is not None)

    :param val: value to check
    """
    return val is not None


def default(val, d):
    """
    Return val if it exists, otherwise return default

    :param val: value to check
    :param d: default value

    """
    return val if exists(val) else d


def cache_fn(f):
    """
    Cache function with argument-based caching

    :param f: function to cache
    """
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


class PreNorm(torch.nn.Module):
    """
    Pre-normalization wrapper module
    """

    def __init__(self, dim, fn, context_dim=None):
        """
        Initialize PreNorm module

        :param dim: input dimension
        :param fn: function to apply
        :param context_dim: context dimension
        """
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = (
            torch.nn.LayerNorm(context_dim) if exists(context_dim) else None
        )

    def forward(self, x, **kwargs):
        """
        Forward pass

        :param x: input tensor
        :param kwargs: additional keyword arguments
        :return: output tensor
        """
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs["context"]
            if context is not None:
                normed_context = self.norm_context(context)
                kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(torch.nn.Module):
    """
    Gated Linear Unit with GELU activation
    """

    def forward(self, x):
        """
        Forward pass

        :param x: input tensor
        :return: output tensor
        """
        x, gates = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gates)


class FeedForward(torch.nn.Module):
    """
    Feed-forward network with optional DropPath
    """

    def __init__(self, dim, mult=4, drop_path_rate=0.0):
        """
        Initialize FeedForward module

        :param dim: input dimension
        :param mult: multiplier
        :param drop_path_rate: drop path rate
        """
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim),
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else torch.nn.Identity()
        )

    def forward(self, x):
        """
        Forward pass

        :param x: input tensor
        :return: output tensor
        """
        return self.drop_path(self.net(x))


class Attention(torch.nn.Module):
    """
    Multi-head attention module
    """

    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0
    ):
        """
        Initialize Attention module

        :param query_dim: query dimension
        :param context_dim: context dimension
        :param heads: number of heads
        :param dim_head: head dimension
        :param drop_path_rate: drop path rate
        """
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = torch.nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else torch.nn.Identity()
        )

    def forward(self, x, context=None, mask=None):
        """
        Forward pass

        :param x: input tensor
        :param context: context tensor
        :param mask: mask tensor
        :return: output tensor
        """
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # Reshape for multi-head attention
        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        q, k, v = (rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q, k, v))

        # Compute attention scores
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # Attention (what we cannot get enough of)...
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)

        # Reshape and project out
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.drop_path(self.to_out(out))


class PointEmbed(torch.nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )
        self.register_buffer("basis", e)  # 3 x 16

        self.mlp = torch.nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(
            torch.cat([self.embed(input, self.basis), input], dim=2)
        )  # B x N x C
        return embed


class DiagonalGaussianDistribution:
    """
    Represents a multivariate Gaussian distribution with a diagonal covariance matrix.

    Attributes:
        mean (torch.Tensor): The mean tensor of the distribution.
        logvar (torch.Tensor): The logarithm of the variance tensor.
        std (torch.Tensor): The standard deviation tensor.
        var (torch.Tensor): The variance tensor.
        deterministic (bool): Flag indicating if the distribution is deterministic.
    """

    def __init__(self, mean, logvar, deterministic=False):
        """
        Initializes the DiagonalGaussianDistribution instance.

        Args:
            mean (torch.Tensor): The mean tensor of the distribution.
            logvar (torch.Tensor): The logarithm of the variance tensor.
            deterministic (bool, optional): If True, sets the distribution to be deterministic. Defaults to False.
        """
        # Ensure mean and logvar have the same shape
        if mean.shape != logvar.shape:
            raise ValueError("Mean and log variance must have the same shape.")

        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(
            self.logvar, -30.0, 20.0
        )  # Clamping to prevent numerical issues
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.mean.device
            )

    def sample(self):
        """
        Generates a sample from the Gaussian distribution.

        Returns:
            torch.Tensor: A tensor sampled from the distribution, with the same shape as `mean`.
        """
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.mean.device
        )
        return x

    def kl(self, other=None):
        """
        Computes the Kullback-Leibler (KL) divergence between this distribution and another Gaussian distribution.

        Args:
            other (DiagonalGaussianDistribution, optional): Another distribution to compute KL divergence with.
                If None, computes the KL divergence with respect to the standard normal distribution.

        Returns:
            torch.Tensor: A tensor containing the KL divergence for each sample in the batch.
        """
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2]
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        """
        Computes the negative log-likelihood (NLL) of a given sample under the distribution.

        Args:
            sample (torch.Tensor): The sample tensor for which to compute the NLL.

        Returns:
            torch.Tensor: A tensor containing the NLL for each sample in the batch.
        """
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        """
        Returns the mode (mean) of the distribution.

        Returns:
            torch.Tensor: The mean tensor of the distribution.
        """
        return self.mean


class Shape2VecSetAE(BaseEncoderDecoder):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
        density_grid_points=128,
        ckpt_path=None,
        norm_factor=0.05,
        layer_norm_encoding=False,
        clamp_encoding=False,
        activation_fn_encoder=None,
        activation_fn_decoder=None,
        **kwargs,
    ):
        super().__init__()
        print(
            "Normalization: factor:",
            norm_factor,
            "| clamp:",
            clamp_encoding,
            "| layer norm: ",
            layer_norm_encoding,
        )

        self.dim = dim
        self.depth = depth
        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.norm_factor = norm_factor
        self.cross_attend_blocks = torch.nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
        )
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = torch.nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        get_latent_attn(**cache_args),
                        get_latent_ff(**cache_args),
                    ]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )

        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            torch.nn.Linear(queries_dim, output_dim)
            if exists(output_dim)
            else torch.nn.Identity()
        )

        # Design grid query points
        self.density = density_grid_points
        x = np.linspace(-1, 1, self.density + 1)
        y = np.linspace(-1, 1, self.density + 1)
        z = np.linspace(-1, 1, self.density + 1)
        xv, yv, zv = np.meshgrid(x, y, z)
        self.grid_query_points = (  # Shape [1, (self.density + 1)^3, 3]
            torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32))
            .view(3, -1)
            .transpose(0, 1)[None]
            .to("cuda", non_blocking=True)
        ) * 0.5  # query points must be in [-0.5, 0.5]

        self.activation_fn_encoder = activation_fn_encoder
        self.activation_fn_decoder = activation_fn_decoder

        self.clamp_encoding = clamp_encoding
        self.layer_norm_encoding = layer_norm_encoding
        if self.layer_norm_encoding:
            self.layer_norm = torch.torch.nn.LayerNorm([dim])

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt_path):
        """
        Overwriting super method because special case for pth
        """
        if ckpt_path.endswith("pth"):
            self.load_state_dict(torch.load(ckpt_path)["model"])
            print("Loaded checkpoint from", ckpt_path)
        else:
            super().load_checkpoint(ckpt_path)

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs, f"Point cloud has {N}, expecting {self.num_inputs}"

        ###### Subsampling of the point cloud to serve as query: X0 = FPS(X)
        flattened = pc.view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)
        pos = flattened
        ratio = 0.25  # in CLAY the ratio is 0.25; in shape2vecset it is 1.0 * self.num_latents / self.num_inputs
        idx = fps(pos, batch, ratio=ratio)
        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)
        pc_embeddings = self.point_embed(pc)
        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None)
            + sampled_pc_embeddings
        )
        x = cross_ff(x) + x

        if self.layer_norm_encoding:
            x = self.layer_norm(x)

        # clamp values to avoid outliers
        if self.clamp_encoding:
            x = torch.clamp(x, -3, 3)

        if self.activation_fn_encoder is not None:
            x = self.activation_fn_encoder(x)

        return x

    def decode(self, x, queries):

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # Cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        # Optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        decoded = self.to_outputs(latents)

        # Apply activation function - sigmoid is suitable since used in loss function
        if self.activation_fn_decoder is not None:
            decoded = self.activation_fn_decoder(decoded)

        return decoded

    def encode_wrapper(self, batch):
        return self.encode(batch["pcl"]) * self.norm_factor

    def decode_wrapper(self, encoded_rep, query_points=None):
        encoded_rep = encoded_rep / self.norm_factor
        # Decode at grid query points
        if query_points is None:
            query_points = self.grid_query_points  # Shape [1, (self.density + 1)^3, 3]

        bs = encoded_rep.size()[0]
        decoded = (
            self.decode(encoded_rep, query_points)
            .view(bs, self.density + 1, self.density + 1, self.density + 1)
            .permute(0, 3, 1, 2)
        )
        return decoded

    def forward(self, batch):
        x = self.encode(batch["pcl"])
        o = self.decode(x, batch["query_points"]).squeeze(-1)
        return {"logits": o}


class KLShape2VecSetAE(torch.nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
        density_grid_points=128,
        ckpt_path=None,
        norm_factor=0.05,
        layer_norm_encoding=False,
        clamp_encoding=False,
        activation_fn_encoder=None,
        activation_fn_decoder=None,
        **kwargs,
    ):
        super().__init__()
        print(
            "Normalization: factor:",
            norm_factor,
            "| clamp:",
            clamp_encoding,
            "| layer norm: ",
            layer_norm_encoding,
        )

        self.depth = depth
        self.num_inputs = num_inputs
        self.num_latents = num_latents
        self.norm_factor = norm_factor
        self.cross_attend_blocks = torch.nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )
        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
        )
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = torch.nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        get_latent_attn(**cache_args),
                        get_latent_ff(**cache_args),
                    ]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            torch.nn.Linear(queries_dim, output_dim)
            if exists(output_dim)
            else torch.nn.Identity()
        )

        self.proj = torch.nn.Linear(latent_dim, dim)
        self.mean_fc = torch.nn.Linear(dim, latent_dim)
        self.logvar_fc = torch.nn.Linear(dim, latent_dim)

        # Design grid query points
        self.density = density_grid_points
        x = np.linspace(-1, 1, self.density + 1)
        y = np.linspace(-1, 1, self.density + 1)
        z = np.linspace(-1, 1, self.density + 1)
        xv, yv, zv = np.meshgrid(x, y, z)
        self.grid_query_points = (  # Shape [1, (self.density + 1)^3, 3]
            torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32))
            .view(3, -1)
            .transpose(0, 1)[None]
            .to("cuda", non_blocking=True)
        ) * 0.5  # query points must be in [-0.5, 0.5]

        self.activation_fn_encoder = activation_fn_encoder
        self.activation_fn_decoder = activation_fn_decoder

        self.clamp_encoding = clamp_encoding
        self.layer_norm_encoding = layer_norm_encoding
        if self.layer_norm_encoding:
            self.layer_norm = torch.torch.nn.LayerNorm([dim])

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt_path):
        """
        Overwriting super method because special case for pth
        """
        if ckpt_path.endswith("pth"):
            self.load_state_dict(torch.load(ckpt_path)["model"])
            print("Loaded checkpoint from", ckpt_path)
        else:
            super().load_checkpoint(ckpt_path)

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs, f"Point cloud has {N}, expecting {self.num_inputs}"

        ###### Subsampling of the point cloud to serve as query: X0 = FPS(X)
        flattened = pc.view(B * N, D)
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)
        pos = flattened
        ratio = 0.25  # in CLAY the ratio is 0.25; in shape2vecset it is 1.0 * self.num_latents / self.num_inputs
        idx = fps(pos, batch, ratio=ratio)
        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)
        pc_embeddings = self.point_embed(pc)
        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None)
            + sampled_pc_embeddings
        )
        x = cross_ff(x) + x

        if self.layer_norm_encoding:
            x = self.layer_norm(x)

        # Extra normalization
        if self.clamp_encoding:
            x = torch.clamp(x, -3, 3)

        if self.activation_fn_encoder is not None:
            x = self.activation_fn_encoder(x)

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x

    def decode(self, x, queries):

        x = self.proj(x)
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # Cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        # Optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        decoded = self.to_outputs(latents)

        # Apply activation function - sigmoid is suitable since used in loss function
        if self.activation_fn_decoder is not None:
            decoded = self.activation_fn_decoder(decoded)

        return decoded

    def encode_wrapper(self, batch):
        _, x = self.encode(batch["pcl"])
        return x * self.norm_factor

    def decode_wrapper(self, encoded_rep, query_points=None):
        encoded_rep = encoded_rep / self.norm_factor
        # Decode at grid query points
        if query_points is None:
            query_points = self.grid_query_points  # Shape [1, (self.density + 1)^3, 3]

        bs = encoded_rep.size()[0]
        decoded = (
            self.decode(encoded_rep, query_points)
            .view(bs, self.density + 1, self.density + 1, self.density + 1)
            .permute(0, 3, 1, 2)
        )
        return decoded

    def forward(self, batch):
        kl, x = self.encode(batch["pcl"])
        o = self.decode(x, batch["query_points"]).squeeze(-1)
        return {"logits": o, "kl": kl}
