import torch
import torch.nn.functional as F

def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token

def sample(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, temperature: float = 1.0) -> torch.Tensor:
    # 应用温度缩放
    if temperature != 1.0:
        logits = logits / temperature
    
    # 处理top-k采样
    if top_k > 0:
        # 对于每个样本，保留top-k个最大的logits，其余设为负无穷
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, :, -1:]  # 获取第k大的值
        logits = torch.where(logits < min_values, float('-inf'), logits)
    
    # 处理top-p采样
    if top_p < 1.0:
        # 对logits进行softmax转换为概率分布
        probs = F.softmax(logits, dim=-1)
        # 对概率进行排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # 创建掩码，保留累积概率小于等于top_p的token
        mask = cumulative_probs <= top_p
        # 将掩码左移一位，确保至少保留一个token
        mask = torch.roll(mask, 1, dims=-1)
        mask[:, :, 0] = True  # 确保第一个token总是被保留
        # 创建一个全为负无穷的新logits张量
        filtered_logits = torch.full_like(logits, float('-inf'))
        # 使用原始logits填充保留的位置
        batch_indices = torch.arange(logits.shape[0]).unsqueeze(1).unsqueeze(2)
        time_indices = torch.arange(logits.shape[1]).unsqueeze(0).unsqueeze(2)
        filtered_logits[batch_indices, time_indices, sorted_indices] = torch.where(
            mask, sorted_probs, float('-inf')
        )
        logits = filtered_logits
    
    # 应用softmax获取概率分布
    probs = F.softmax(logits, dim=-1)
    # 采样
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).reshape(probs.shape[:-1])


def detect_repetition(text, min_repeats=3, window_size=5, max_distance=10):
    import re
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    words = re.findall(r'\w+', cleaned_text)

    ngrams = [' '.join(words[i:i+window_size]) for i in range(len(words) - window_size + 1)]

    seen = {}
    for index, ngram in enumerate(ngrams):
        if ngram in seen:
            if index - seen[ngram][-1] <= max_distance:
                seen[ngram].append(index)
                if len(seen[ngram]) >= min_repeats:
                    return True
            else:
                seen[ngram] = [index]
        else:
            seen[ngram] = [index]
    
    return False


def amo_sampling(sample, sigma, sigma_next, pred_v, overshoot=3):
    # Upcast to avoid precision issues when computing prev_sample
    t = sigma
    s = sigma_next
    x_t = sample
    c = overshoot  # 2
    o = min(s + c * (s - t), 1)
    pred_x_o = x_t + (o - t) * pred_v
    a = s / o
    b = (torch.clamp_min((1 - s) ** 2 - (a * (1 - o)) ** 2, 0)) ** 0.5
    noises = torch.randn(size=x_t.shape, device=x_t.device)
    prev_sample = a * pred_x_o + b * noises
    prev_sample = prev_sample.to(pred_v.dtype)
    return prev_sample


def stochastic_round(x):
    x_floor = torch.floor(x)
    frac = (x - x_floor).clamp(0, 1)
    x = (x_floor + torch.bernoulli(frac)).to(torch.long)
    return x


if __name__ == "__main__":
    # 创建一个示例logits张量 [B=1, T=1, C=5]
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    
    # 采样参数
    top_k = 3
    top_p = 0.9
    temperature = 0.8
    
    # 执行采样
    sampled_token = sample(logits, top_k, top_p, temperature)
    print('sampled_token', sampled_token)
