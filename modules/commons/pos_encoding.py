import math
import torch
import torch.nn as nn

class SinusoidalPositionalEmbedding(nn.Module):
    """
    pos_emb = SinusoidalPositionalEmbedding(embedding_dim=512, init_size=1024)
    
    # 单个位置（标量 timestep），返回 (D,)
    e5 = pos_emb(timestep=5)             # torch.Size([512])
    
    # 一维 positions（序列），返回 (T, D)
    positions = torch.arange(0, 16)      # [0..15]
    e_seq = pos_emb(positions=positions) # torch.Size([16, 512])
    
    # 二维 positions（如(B, T)），返回 (B, T, D)
    batch_pos = torch.stack([positions, positions], dim=0) # (2, 16)
    e_batch = pos_emb(positions=batch_pos)                 # torch.Size([2, 16, 512])
    
    # 1-based 索引输入（例如某些外部系统），用 offset=-1 调整为 0-based
    pos_1_based = torch.tensor([1, 2, 3])
    e_1_based = pos_emb(positions=pos_1_based, offset=-1)  # 取实际 [0,1,2]
    """
    def __init__(self, embedding_dim, proj=True, init_size=1024, store_dtype=torch.float32):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError(f"embedding_dim must be >= 2, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.weights = self.get_embedding(
            num_embeddings=init_size,
            embedding_dim=embedding_dim,
            device=None,
            dtype=store_dtype,
        )
        if proj:
            self.proj = nn.Linear(embedding_dim, embedding_dim)
        else:
            self.proj = nn.Identity()
        
    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, device=None, dtype=torch.float32):
        half_dim = embedding_dim // 2
        if half_dim > 1:
            div_term = torch.exp(
                torch.arange(half_dim, device=device, dtype=dtype)
                * (-math.log(10000.0) / (half_dim - 1))
            )
        elif half_dim == 1:
            div_term = torch.ones(1, device=device, dtype=dtype)
        else:
            raise ValueError("Invalid embedding_dim leading to half_dim == 0")

        positions = torch.arange(num_embeddings, device=device, dtype=dtype).unsqueeze(1)  # [L, 1]
        angles = positions * div_term.unsqueeze(0)  # [L, half_dim]

        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [L, 2*half_dim]
        if embedding_dim % 2 == 1:
            pad = torch.zeros(num_embeddings, 1, device=device, dtype=dtype)
            emb = torch.cat([emb, pad], dim=1)  # [L, D]
        return emb

    @torch.no_grad()
    def _maybe_grow(self, max_pos_needed):
        if max_pos_needed > self.weights.size(0):
            self.weights = self.get_embedding(
                num_embeddings=max_pos_needed,
                embedding_dim=self.embedding_dim,
                device=self.weights.device,
                dtype=self.weights.dtype,
            )

    def forward(self, timestep=None, positions=None, offset=0, out_dtype=None, device=None):
        if (timestep is None) and (positions is None):
            raise ValueError("Either `timestep` or `positions` must be provided.")

        dev = self.weights.device if device is None else device
        self.weights = self.weights.to(device=dev)
        store_dtype = self.weights.dtype

        max_index = 0
        if timestep is not None:
            if torch.is_tensor(timestep):
                if timestep.numel() == 0:
                    raise ValueError("Empty `timestep` tensor.")
                tmax = int(timestep.max().item())
            else:
                tmax = int(timestep)
            max_index = max(max_index, tmax + offset)

        if positions is not None:
            if not torch.is_tensor(positions):
                raise TypeError("`positions` must be a torch.Tensor.")
            if positions.numel() == 0:
                raise ValueError("Empty `positions` tensor.")
            pmax = int(positions.max().item())
            max_index = max(max_index, pmax + offset)

        if max_index < 0:
            raise ValueError("All (index + offset) must be non-negative.")

        max_pos_needed = max_index + 1
        self._maybe_grow(max_pos_needed)

        out_dtype = out_dtype or store_dtype

        if positions is not None:
            idx = positions.to(device=dev, dtype=torch.long) + offset
            if (idx < 0).any():
                raise ValueError("positions + offset contains negative indices.")
            emb = self.weights.index_select(0, idx.reshape(-1))
            emb = emb.reshape(*positions.shape, -1)
            return self.proj(emb.to(dtype=out_dtype))

        if torch.is_tensor(timestep):
            idx = timestep.to(device=dev, dtype=torch.long) + offset
            if (idx < 0).any():
                raise ValueError("timestep + offset contains negative indices.")
            emb = self.weights.index_select(0, idx.reshape(-1)).reshape(*idx.shape, -1)
            return self.proj(emb.to(dtype=out_dtype))
        else:
            idx = int(timestep) + offset
            if idx < 0:
                raise ValueError("timestep + offset is negative.")
            return self.proj(self.weights[idx].to(dtype=out_dtype))  # (D,)


if __name__ == '__main__':
    hard_pos_embed = SinusoidalPositionalEmbedding(1024)
    
    txt_embeds = torch.rand((1, 10, 1024))
    pos_embed = hard_pos_embed(
                torch.arange(txt_embeds.shape[1], device=txt_embeds.device), 
                out_dtype=txt_embeds.dtype, device=txt_embeds.device
            )
    print(pos_embed.shape)
    print((txt_embeds + pos_embed).shape)
    print(f"{txt_embeds = }")
    print(f"{pos_embed = }")
    print(f"{txt_embeds + pos_embed = }")
    