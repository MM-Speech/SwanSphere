import torch
import torch.nn.functional as F

_warned = set()
def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(msg)

def _maybe_fp32(x, cond: bool):
    return x.float() if cond and x.dtype in (torch.float16, torch.bfloat16) else x

@torch.no_grad()
def _stats_for_ln(x_fp32, eps):
    mean = x_fp32.mean(dim=-1, keepdim=True)
    var = x_fp32.var(dim=-1, unbiased=False, keepdim=True)
    rstd = (var + eps).rsqrt()
    return mean, rstd

def layer_norm_fn(
    x,
    weight=None,
    bias=None,
    eps: float = 1e-5,
    residual=None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    return_mean: bool = False,
    return_rstd: bool = False,
    out_dtype=None,
    **kwargs,
):
    """
    纯 PyTorch LN 兜底实现，尽量兼容 flash_attn 的 layer_norm_fn 常见调用。
    行为：
      - y = LayerNorm(x)
      - 若 residual is not None: y += residual
      - 忽略 prenorm（多数场景不依赖该标志控制分支）
      - residual_in_fp32=True 时，在 fp32 中计算，再 cast 回输入 dtype
      - return_mean/return_rstd 时返回附加统计量
    可能的细微差异：
      - 与 flash-attn 融合核的数值略有不同（允许的可接受范围内）
      - prenorm 标志未参与分支（若你的调用强依赖该标志，请在上层封装）
    """
    if prenorm:
        _warn_once("prenorm", "[fa_fallbacks] prenorm is ignored in fallbacks (y=LN(x) (+ residual))。")

    compute_in_fp32 = residual_in_fp32
    x_comp = _maybe_fp32(x, compute_in_fp32)
    w = weight.to(x_comp.dtype) if weight is not None else None
    b = bias.to(x_comp.dtype) if bias is not None else None

    y = F.layer_norm(x_comp, x_comp.shape[-1:], w, b, eps)

    if residual is not None:
        res = residual.to(x_comp.dtype) if compute_in_fp32 else residual
        y = y + res

    out_dtype = out_dtype or x.dtype
    y = y.to(out_dtype)

    if return_mean or return_rstd:
        with torch.no_grad():
            mean, rstd = _stats_for_ln(x_comp, eps)
            mean = mean.squeeze(-1).to(out_dtype)
            rstd = rstd.squeeze(-1).to(out_dtype)

        outs = [y]
        if return_mean:
            outs.append(mean)
        if return_rstd:
            outs.append(rstd)
        return tuple(outs)

    return y


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size)) if bias else None

    def forward(
        self,
        x,
        residual=None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        out_dtype=None,
        **kwargs,  # 兼容多余形参
    ):
        if prenorm:
            _warn_once("rms_prenorm", "[fa_fallbacks] RMSNorm 的 prenorm 标志在兜底实现中被忽略。")

        compute_in_fp32 = residual_in_fp32
        x_comp = _maybe_fp32(x, compute_in_fp32)

        # RMSNorm: x / sqrt(mean(x^2) + eps)
        rms = x_comp.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        y = x_comp * rms

        if residual is not None:
            res = residual.to(x_comp.dtype) if compute_in_fp32 else residual
            y = y + res

        y = y.to(self.weight.dtype) * self.weight
        if self.bias is not None:
            y = y + self.bias

        out_dtype = out_dtype or x.dtype
        return y.to(out_dtype)
    
    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        if hasattr(self, 'bias'):
            torch.nn.init.zeros_(self.bias)
