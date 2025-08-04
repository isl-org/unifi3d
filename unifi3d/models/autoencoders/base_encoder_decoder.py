from abc import ABC, abstractmethod
import torch.nn as nn
from unifi3d.utils.model.model_utils import load_net_from_safetensors
import torch


class BaseEncoderDecoder(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        self._compute_model_size()

    @abstractmethod
    def encode(self, input_data) -> None:
        """
        Encode the input data to its latent representation.
        Args:
            input_data (torch.Tensor): The input data to encode.

        Returns:
            torch.Tensor: The encoded latent representation of the input data.
        """
        pass

    @abstractmethod
    def decode(self, latent_representation) -> None:
        """
        Decode the latent representation back to the input space.
        Args:
            latent_representation (torch.Tensor): The latent representation to decode.

        Returns:
            torch.Tensor: The decoded output data.
        """
        pass

    def forward(self, input_data) -> None:
        """
        A forward pass through the model, by default just encoding and decoding the data
        Args:
            input_data (torch.Tensor): The input data to process.

        Returns:
            torch.Tensor: The reconstructed data after encoding and decoding.
        """
        encoded = self.encode(input_data)
        decoded = self.decode(encoded)
        return decoded

    @abstractmethod
    def encode_wrapper(self, data_sample, **kwargs) -> None:
        """
        Encode the batch given as a dictionary.
        Args:
            data_sample (dict): A dictionary containing the batch data.
        Returns:
            torch.Tensor or tuple of tensors: The encoded representations.
        """
        pass

    @abstractmethod
    def decode_wrapper(self, encoded, query_points=None):
        """
        Decode the latents of a batch.
        Args:
            encoded (torch.Tensor or tuple of tensors): latent representation of batch
            query_points (torch.Tensor): The query points to decode the latent
                representation to. If None is passed, it's expected to be set
                (e.g., uniform) in the decode_wrapper function
        Returns:
            torch.Tensor or tuple of tensors: The decoded data.
        """
        pass

    def _compute_model_size(self):
        """
        Computes and stores the total size (in bytes) of the encoder-decoder model (self.encoder_decoder),
        including both their parameters and buffers.

        The computed size is stored in self.total_size.
        """
        param_size = sum(p.element_size() * p.numel() for p in self.parameters())
        buffer_size = sum(b.element_size() * b.numel() for b in self.buffers())
        self.total_size = param_size + buffer_size  # in bytes

    def load_checkpoint(self, ckpt_path):
        """
        Load a checkpoint from a file. Supports safetensors, pth and pt
        """
        # load checkpoint
        if ckpt_path.endswith("safetensors"):
            state_dict = load_net_from_safetensors(ckpt_path)
        elif ckpt_path.endswith("pt") or ckpt_path.endswith("pth"):
            state_dict = torch.load(ckpt_path)
        else:
            raise RuntimeError(f"File format not recognized for {ckpt_path}")
        self.load_state_dict(state_dict)
        print("Loaded checkpoint from", ckpt_path)
