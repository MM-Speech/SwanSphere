import torch
import torch.nn as nn

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        return self.proj(x)


class SiglipEncoder(nn.Module):
    def __init__(self, out_channels, pretrained_model_name_or_path='google/siglip-so400m-patch14-384', use_pooler_output=True):
        super().__init__()

        from transformers import SiglipVisionModel, SiglipImageProcessor

        self.encoder = SiglipVisionModel.from_pretrained(pretrained_model_name_or_path)
        for param in self.encoder.parameters():
            param.requires_grad = False
            param.grad = None

        self.out_channels = out_channels
        self.use_pooler_output = use_pooler_output
        self.out_proj = nn.Linear(self.encoder.config.hidden_size, out_channels)

    def forward(self, x):
        # x [B, 3, 384, 384]
        with torch.no_grad():
            outputs = self.encoder(pixel_values=x)
        if self.use_pooler_output:
            x = outputs.pooler_output    # [B, 1152]
        else:
            x = outputs.last_hidden_state    # [B, 729, 1152]

        x = self.out_proj(x)
        return x
