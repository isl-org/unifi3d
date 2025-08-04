"""
MIT License

Copyright (c) 2023 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional
from typing import OrderedDict, Sequence, TypeVar, Union

import torch
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np


ArrayType = Union[np.ndarray, Iterable[int], torch.Tensor]

K = TypeVar("K")
V = TypeVar("V")


class AttrDict(OrderedDict[K, V], Generic[K, V]):
    """
    An attribute dictionary that automatically handles nested keys joined by "/".

    Originally copied from: https://stackoverflow.com/questions/3031219/recursively-access-dict-via-attributes-as-well-as-index-access
    """

    MARKER = object()

    # pylint: disable=super-init-not-called
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            for key, value in kwargs.items():
                self.__setitem__(key, value)
        else:
            assert len(args) == 1
            assert isinstance(args[0], (dict, AttrDict))
            for key, value in args[0].items():
                self.__setitem__(key, value)

    def __contains__(self, key):
        if "/" in key:
            keys = key.split("/")
            key, next_key = keys[0], "/".join(keys[1:])
            return key in self and next_key in self[key]
        return super(AttrDict, self).__contains__(key)

    def __setitem__(self, key, value):
        if "/" in key:
            keys = key.split("/")
            key, next_key = keys[0], "/".join(keys[1:])
            if key not in self:
                self[key] = AttrDict()
            self[key].__setitem__(next_key, value)
            return

        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(**value)
        if isinstance(value, list):
            value = [AttrDict(val) if isinstance(val, dict) else val for val in value]
        super(AttrDict, self).__setitem__(key, value)

    def __getitem__(self, key):
        if "/" in key:
            keys = key.split("/")
            key, next_key = keys[0], "/".join(keys[1:])
            val = self[key]
            if not isinstance(val, AttrDict):
                raise ValueError
            return val.__getitem__(next_key)

        return self.get(key, None)

    def all_keys(
        self,
        leaves_only: bool = False,
        parent: Optional[str] = None,
    ) -> List[str]:
        keys = []
        for key in self.keys():
            cur = key if parent is None else f"{parent}/{key}"
            if not leaves_only or not isinstance(self[key], dict):
                keys.append(cur)
            if isinstance(self[key], dict):
                keys.extend(self[key].all_keys(leaves_only=leaves_only, parent=cur))
        return keys

    def dumpable(self, strip=True):
        """
        Casts into OrderedDict and removes internal attributes
        """

        def _dump(val):
            if isinstance(val, AttrDict):
                return val.dumpable()
            elif isinstance(val, list):
                return [_dump(v) for v in val]
            return val

        if strip:
            return {k: _dump(v) for k, v in self.items() if not k.startswith("_")}
        return {
            k: _dump(v if not k.startswith("_") else repr(v)) for k, v in self.items()
        }

    def map(
        self,
        map_fn: Callable[[Any, Any], Any],
        should_map: Optional[Callable[[Any, Any], bool]] = None,
    ) -> "AttrDict":
        """
        Creates a copy of self where some or all values are transformed by
        map_fn.

        :param should_map: If provided, only those values that evaluate to true
            are converted; otherwise, all values are mapped.
        """

        def _apply(key, val):
            if isinstance(val, AttrDict):
                return val.map(map_fn, should_map)
            elif should_map is None or should_map(key, val):
                return map_fn(key, val)
            return val

        return AttrDict({k: _apply(k, v) for k, v in self.items()})

    def __eq__(self, other):
        return self.keys() == other.keys() and all(
            self[k] == other[k] for k in self.keys()
        )

    def combine(
        self,
        other: Dict[str, Any],
        combine_fn: Callable[[Optional[Any], Optional[Any]], Any],
    ) -> "AttrDict":
        """
        Some values may be missing, but the dictionary structures must be the
        same.

        :param combine_fn: a (possibly non-commutative) function to combine the
            values
        """

        def _apply(val, other_val):
            if val is not None and isinstance(val, AttrDict):
                assert isinstance(other_val, AttrDict)
                return val.combine(other_val, combine_fn)
            return combine_fn(val, other_val)

        # TODO nit: this changes the ordering..
        keys = self.keys() | other.keys()
        return AttrDict({k: _apply(self[k], other[k]) for k in keys})

    __setattr__, __getattr__ = __setitem__, __getitem__


@dataclass
class Query:
    # Both of these are of shape [batch_size x ... x 3]
    position: torch.Tensor
    direction: Optional[torch.Tensor] = None

    t_min: Optional[torch.Tensor] = None
    t_max: Optional[torch.Tensor] = None

    def copy(self) -> "Query":
        return Query(
            position=self.position,
            direction=self.direction,
            t_min=self.t_min,
            t_max=self.t_max,
        )

    def map_tensors(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "Query":
        return Query(
            position=f(self.position),
            direction=f(self.direction) if self.direction is not None else None,
            t_min=f(self.t_min) if self.t_min is not None else None,
            t_max=f(self.t_max) if self.t_max is not None else None,
        )


def to_torch(arr: ArrayType, dtype=torch.float):
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.from_numpy(np.array(arr)).to(dtype)


def sample_pmf(pmf: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Sample from the given discrete probability distribution with replacement.

    The i-th bin is assumed to have mass pmf[i].

    :param pmf: [batch_size, *shape, n_samples, 1] where (pmf.sum(dim=-2) == 1).all()
    :param n_samples: number of samples

    :return: indices sampled with replacement
    """

    *shape, support_size, last_dim = pmf.shape
    assert last_dim == 1

    cdf = torch.cumsum(pmf.view(-1, support_size), dim=1)
    inds = torch.searchsorted(
        cdf, torch.rand(cdf.shape[0], n_samples, device=cdf.device)
    )

    return inds.view(*shape, n_samples, 1).clamp(0, support_size - 1)


def safe_divide(a, b, epsilon=1e-6):
    return a / torch.where(b < 0, b - epsilon, b + epsilon)


def append_tensor(
    val_list: Optional[List[torch.Tensor]], output: Optional[torch.Tensor]
):
    if val_list is None:
        return [output]
    return val_list + [output]


def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.length = length
        input_tensors = list(args[:length])
        input_params = list(args[length:])
        ctx.save_for_backward(*input_tensors, *input_params)
        with torch.no_grad():
            output_tensors = ctx.run_function(*input_tensors)
        return output_tensors

    @staticmethod
    @custom_bwd
    def backward(ctx, *output_grads):
        inputs = ctx.saved_tensors
        input_tensors = inputs[: ctx.length]
        input_params = inputs[ctx.length :]
        res = CheckpointFunctionGradFunction.apply(
            ctx.run_function,
            len(input_tensors),
            len(input_params),
            *input_tensors,
            *input_params,
            *output_grads,
        )
        return (None, None) + res


class CheckpointFunctionGradFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, run_function, length_1, length_2, *args):
        ctx.run_function = run_function
        ctx.length_1 = length_1
        ctx.length_2 = length_2
        input_tensors = [x.detach().requires_grad_(True) for x in args[:length_1]]
        input_params = list(args[length_1 : length_1 + length_2])
        output_grads = list(args[length_1 + length_2 :])
        ctx.save_for_backward(*input_tensors, *input_params, *output_grads)

        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            input_tensors + input_params,
            output_grads,
            allow_unused=True,
        )
        return input_grads

    @staticmethod
    @custom_bwd
    def backward(ctx, *all_output_grads):
        args = ctx.saved_tensors
        input_tensors = [x.detach().requires_grad_(True) for x in args[: ctx.length_1]]
        input_params = list(args[ctx.length_1 : ctx.length_1 + ctx.length_2])
        output_grads = [
            x.detach().requires_grad_(True) for x in args[ctx.length_1 + ctx.length_2 :]
        ]

        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
            input_grads = torch.autograd.grad(
                output_tensors,
                input_tensors + input_params,
                output_grads,
                allow_unused=True,
                create_graph=True,
                retain_graph=True,
            )
        input_grads_grads = torch.autograd.grad(
            input_grads,
            input_tensors + input_params + output_grads,
            all_output_grads,
            allow_unused=True,
        )
        del input_grads
        return (None, None, None) + input_grads_grads
