import torch
import math
import os
import warnings
import numpy as np
import torch.nn as nn
from einops import repeat
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
from PIL import Image


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class ClipTextEncoder(nn.Module):
    def __init__(self, enc_dim, clip_channels=512):
        super().__init__()
        # add CLIP text encoder
        # (loading yields warning, but only because we ignore the weights of the
        # vision encoder)
        self.clip_enc_model = CLIPTextModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_enc_model.parameters():
            param.requires_grad = False
        warnings.warn("NOTE: CLIP layers are frozen")

        self.text_embed_layers = nn.Sequential(
            nn.Linear(clip_channels, enc_dim, bias=True),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim, bias=True),
        )

    def forward(self, text: list):
        inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        outputs = self.clip_enc_model(**inputs)
        text_embeds = outputs.text_embeds
        return self.text_embed_layers(text_embeds)


class ClipImageEncoder(nn.Module):
    def __init__(self, enc_dim, clip_channels=512):
        super().__init__()
        # load CLIP image encoder
        # (loading yields warning, but only because we ignore the weights of the
        # vision encoder)
        self.clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).cuda()
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_vision_model.parameters():
            param.requires_grad = False
        warnings.warn("NOTE: CLIP layers are frozen")

        self.img_embed_layers = nn.Sequential(
            nn.Linear(clip_channels, enc_dim, bias=True),
            nn.SiLU(),
            nn.Linear(enc_dim, enc_dim, bias=True),
        )

    def forward(self, image_paths: list):
        def make_empty_img():
            return np.zeros((100, 100, 3), dtype=np.uint8)

        images = []
        for img_path in image_paths:
            # if there is no such path: add empty image
            if not os.path.exists(img_path):
                warnings.warn("Image path does not exist. Adding empty image.")
                image = make_empty_img()
            # if it is a directory, pick a random image
            elif os.path.isdir(img_path):
                files_avail = os.listdir(img_path)
                # if no files in directory: add empty image
                if len(files_avail) == 0:
                    warnings.warn("Image dir is empty. Adding empty image.")
                    image = make_empty_img()
                else:
                    img_file = os.path.join(img_path, np.random.choice(files_avail))
                    image = Image.open(img_file)
            else:
                # default case: one img path is given
                image = Image.open(img_path)

            images.append(image)

        # preprocess images
        inputs = self.processor(images=images, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        # embed them with CLIP
        outputs = self.clip_vision_model(**inputs)
        img_embeds = outputs.image_embeds
        # pass through MLP to get right shape
        return self.img_embed_layers(img_embeds)
