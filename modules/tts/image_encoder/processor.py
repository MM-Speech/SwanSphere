from typing import Union
from dataclasses import dataclass, fields
import attrdict

import torch
import numpy as np
import cv2
from einops import rearrange
from PIL import Image

# @dataclass
# class ProcessorConfig:
#     patch_size: Union[int, None] = None
#     aug_image: bool = True
#     pre_patch: bool = True
#     apply_img_ids: bool = True

#     fix_num_patches: bool = True
#     num_patches: int = 3600

class ProcessorConfig(attrdict.AttrDict):
    def __init__(self):
        config = dict(
            patch_size=None,
            aug_image=True,
            crop_image=False,
            pre_patch=True,
            apply_img_ids=True,

            fix_num_patches=True,
            num_patches=3600,

            visual_model_name='simple_encoder',
            visual_model_path=''
        )
        super().__init__(config)

def override_config(config: ProcessorConfig, hparams):
    # for k in [field.name for field in fields(config)]:
    for k in config:
        if k in hparams:
            config[k] = hparams[k]
    return config

@dataclass
class ProcessorOutput:
    image: torch.Tensor = None,
    img_ids: torch.Tensor = None

# img_transforms = None
# processor = None

# def img_processor(image: Image, config: ProcessorConfig):
#     visual_model_name = config.visual_model_name
#     if visual_model_name == 'simple_encoder':
#         return simple_processor(image, config)
#     elif visual_model_name == 'google/siglip-so400m-patch14-384':
#         return siglip_processor(image, config)


# def simple_processor(image: Image, config: ProcessorConfig):
#     image = np.array(image)     # [H, W, C], RGB
#     if config.aug_image:
#         global img_transforms
#         if img_transforms is None:
#             from torchvision.transforms import v2, InterpolationMode
#             img_transforms = v2.Compose([
#                 v2.RandomHorizontalFlip(0.5)
#             ])
#         image = img_transforms(torch.from_numpy(image).permute(2, 0, 1)).permute(1, 2, 0).numpy()
#     if image.dtype != np.float32:
#         image = image.astype(np.float32) / 255
#     if config.pre_patch and config.patch_size is not None:
#         patch_size = config.patch_size
#         h, w, _ = image.shape
#         if config.fix_num_patches:     # (1280, 720) -> (1120, 630) = (80 * 14, 45 * 14); (1920, 1080) -> (1120, 630); (1080, 1080) -> (60 * 14, 60 * 14)
#             resize_ratio = (config.num_patches * patch_size * patch_size / h / w)**0.5
#             new_w, new_h = round(w * resize_ratio), round(h * resize_ratio)
#         else:
#             new_w, new_h = w // patch_size * patch_size, h // patch_size * patch_size
#         image = cv2.resize(image, (new_w, new_h))
#         image = torch.from_numpy(image)     # [H, W, 3]
#         image = rearrange(image, '(h ph) (w pw) c -> (h w) (ph pw c)', ph=patch_size, pw=patch_size)    # [T, C]
#         if config.fix_num_patches:
#             assert image.shape[0] == config.num_patches, image.shape
#         img_ids = None
#         if config.apply_img_ids:
#             img_ids = torch.zeros(new_h // patch_size, new_w // patch_size, 2)
#             img_ids[..., 0] = img_ids[..., 0] + torch.arange(new_h // patch_size)[:, None]
#             img_ids[..., 1] = img_ids[..., 1] + torch.arange(new_w // patch_size)[None, :]
#             img_ids = rearrange(img_ids, "h w c -> (h w) c")    # [T, 2]
#     return ProcessorOutput(
#         image=image,
#         img_ids=img_ids
#     )


# def siglip_processor(image: Image, config: ProcessorConfig):
#     if config.crop_image:
#         w, h = image.size
#         from torchvision.transforms import v2
#         cropper = v2.RandomCrop(size=min(w, h))
#         image = cropper(image)
#     global processor
#     if processor is None:
#         from transformers import SiglipImageProcessor
#         processor = SiglipImageProcessor.from_pretrained(config.visual_model_path)
#     image = processor(images=[image], return_tensors='pt').pixel_values   # [1, 3, 384, 384]
#     return ProcessorOutput(
#         image=image
#     )


class ImageProcessor:
    def __init__(self, config: ProcessorConfig = None, hparams=None):
        if config is not None:
            self.config = config
        self.config = ProcessorConfig()
        if hparams is not None:
            override_config(self.config, hparams)

    def process(self, image: Image.Image):
        config = self.config
        visual_model_name = config.visual_model_name
        if visual_model_name == 'simple_encoder':
            return self.simple_processor(image)
        elif visual_model_name == 'google/siglip-so400m-patch14-384':
            return self.siglip_processor(image)
        
    def __call__(self, image: Image.Image):
        return self.process(image)

    def simple_processor(self, image: Image.Image):
        config = self.config
        image = np.array(image)     # [H, W, C], RGB
        if config.aug_image:
            if not hasattr(self, 'img_transforms') or self.img_transforms is None:
                from torchvision.transforms import v2
                self.img_transforms = v2.Compose([
                    v2.RandomHorizontalFlip(0.5)
                ])
            image = self.img_transforms(torch.from_numpy(image).permute(2, 0, 1)).permute(1, 2, 0).numpy()
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255
        if config.pre_patch and config.patch_size is not None:
            patch_size = config.patch_size
            h, w, _ = image.shape
            if config.fix_num_patches:     # (1280, 720) -> (1120, 630) = (80 * 14, 45 * 14); (1920, 1080) -> (1120, 630); (1080, 1080) -> (60 * 14, 60 * 14)
                resize_ratio = (config.num_patches * patch_size * patch_size / h / w)**0.5
                new_w, new_h = round(w * resize_ratio), round(h * resize_ratio)
            else:
                new_w, new_h = w // patch_size * patch_size, h // patch_size * patch_size
            image = cv2.resize(image, (new_w, new_h))
            image = torch.from_numpy(image)     # [H, W, 3]
            image = rearrange(image, '(h ph) (w pw) c -> (h w) (ph pw c)', ph=patch_size, pw=patch_size)    # [T, C]
            if config.fix_num_patches:
                assert image.shape[0] == config.num_patches, image.shape
            img_ids = None
            if config.apply_img_ids:
                img_ids = torch.zeros(new_h // patch_size, new_w // patch_size, 2)
                img_ids[..., 0] = img_ids[..., 0] + torch.arange(new_h // patch_size)[:, None]
                img_ids[..., 1] = img_ids[..., 1] + torch.arange(new_w // patch_size)[None, :]
                img_ids = rearrange(img_ids, "h w c -> (h w) c")    # [T, 2]
        return ProcessorOutput(
            image=image,
            img_ids=img_ids
        )

    def siglip_processor(self, image: Image.Image):
        config = self.config
        if config.crop_image:
            w, h = image.size
            from torchvision.transforms import v2
            cropper = v2.RandomCrop(size=min(w, h))
            image = cropper(image)
        if not hasattr(self, 'processor') or self.processor is None:
            from transformers import SiglipImageProcessor
            self.processor = SiglipImageProcessor.from_pretrained(config.visual_model_path)
        image = self.processor(images=[image], return_tensors='pt').pixel_values[0]   # [3, 384, 384]
        return ProcessorOutput(
            image=image
        )