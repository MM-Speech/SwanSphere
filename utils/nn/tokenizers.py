from functools import partial

import numpy as np
import torch

def encode_space(values, space='linear', scale=1.0):
    is_tensor = isinstance(values, torch.Tensor)
    if space == 'linear':
        z = values * scale
    elif space == 'log1p':
        if is_tensor:
            z = torch.log1p(values.clamp_min_(0.0)) * scale
        else:
            z = np.log1p(np.clip(values, 0.0, None)) * scale
    else:
        raise NotImplementedError
    return z

def decode_space(z, space='linear', scale=1.0):
    is_tensor = isinstance(z, torch.Tensor)
    if space == 'linear':
        values = z / scale
    elif space == 'log1p':
        if is_tensor:
            values = torch.expm1(z / scale)
        else:
            values = np.expm1(z / scale)
    else:
        raise NotImplementedError
    return values

class QuantTokenizer:
    def __init__(
            self, 
            num_codes,
            min_value,
            max_value,
            underflow_value=None,
            overflow_value=None,
            idx_start=0,
            space='linear',  # lineaer | log1p
            scale=1.0,
            device='cpu',
    ):
        self.space = space
        self.encode_space = partial(encode_space, space=space, scale=scale)
        self.decode_space = partial(decode_space, space=space, scale=scale)

        assert num_codes > 2
        self.num_codes = num_codes
        self.min_value = min_value = self.encode_space(min_value)
        self.max_value = max_value = self.encode_space(max_value)
        self.step = (max_value - min_value) / (num_codes - 2)

        self.idx_start = int(idx_start)
        self.boundaries = torch.linspace(min_value, max_value, num_codes - 1).to(device)
        self.underflow_idx = 0
        self.overflow_idx = num_codes - 1

        self.underflow_value = self.encode_space(underflow_value) if underflow_value is not None else self.min_value
        self.overflow_value = self.encode_space(overflow_value) if overflow_value is not None else self.max_value


    def encode(self, values, return_center=False, return_residual=False):
        z = self.encode_space(values)

        tokens = torch.bucketize(z, self.boundaries) + self.idx_start
        ret = {'tokens': tokens}

        def get_centers():
            centers = (tokens - 1) * self.step + self.min_value + self.step / 2
            centers[tokens == self.underflow_idx] = z[tokens == self.underflow_idx]
            centers[tokens == self.overflow_idx] = z[tokens == self.overflow_idx]
            return centers

        # | -inf | 1 | 2 | 3 | 
        centers = None
        if return_center:
            centers = get_centers()
            ret['centers'] = centers

        if return_residual:
            if centers is None:
                centers = get_centers()
            residuals = z - centers
            ret['residuals'] = residuals
        
        return ret

    def decode(self, tokens, residuals=None):
        centers = (tokens - 1) * self.step + self.min_value + self.step / 2
        centers[tokens == self.underflow_idx] = self.underflow_value
        centers[tokens == self.overflow_idx] = self.overflow_value

        if residuals is not None:
            centers = centers + residuals

        centers = self.decode_space(centers)
        
        return centers


if __name__ == '__main__':
    torch.random.manual_seed(42)

    tokenizer = QuantTokenizer(
        num_codes=64,
        min_value=0,
        max_value=128,
        underflow_value=0,
        overflow_value=128,
        space='log1p',
        # scale=10,
    )
    print('boundaries', tokenizer.boundaries)
    print('boundaries', tokenizer.decode_space(tokenizer.boundaries))
    print('step', tokenizer.step)
    print('min_value', tokenizer.min_value)
    print('max_value', tokenizer.max_value)

    # values = (torch.rand(20) * 128).long().float()
    values = torch.Tensor(list(range(20)) + [70, 80, 90, 100, 110, 120])
    print('values', values)
    ret = tokenizer.encode(values, True, True)
    tokens = ret['tokens']
    centers = ret['centers']
    residuals = ret['residuals']
    print('tokens', tokens)
    print('centers', centers)
    print('residuals', residuals)
    values = tokenizer.decode(tokens)
    print('values', values)
    values = tokenizer.decode(tokens, residuals)
    print('values', values)


