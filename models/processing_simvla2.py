"""SimVLA2 processor.

BERT tokenizer for text and fast torch-based image preprocessing for SigLIP2
(256x256).
"""

from typing import List, Union, Dict, Any, Optional

import os

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import logging

from transformers import AutoTokenizer


# TEXT_ATTN_MASK is not saved into config.json; set it the same way for
# training and evaluation.
def _text_attn_mask_enabled() -> bool:
    return os.environ.get("TEXT_ATTN_MASK", "0") == "1"


class SimVLA2Processor:
    """Unified multimodal processor for SimVLA2.

    - Text: bert-base-uncased tokenizer, max_length=50
    - Image: resize to 256x256, ImageNet normalization
    """

    num_views: int = 3
    image_size: int = 256
    language_max_length: int = 50

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        image_size: int = 256,
        num_views: int = 3,
        language_max_length: int = 50,
    ):
        self.image_size = image_size
        self.num_views = num_views
        self.language_max_length = language_max_length

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logging.info(f"[SimVLA2Processor] tokenizer={tokenizer_name}, "
                     f"vocab_size={self.tokenizer.vocab_size}")

        # ImageNet mean/std, not SigLIP2's native 0.5/0.5. The vision tower is
        # fine-tuned under this normalization, so checkpoints depend on it.
        self._image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def encode_language(self, language_instruction: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]

        inputs = self.tokenizer(
            language_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.language_max_length,
            truncation=True,
        )
        out = {"input_ids": inputs["input_ids"]}
        if _text_attn_mask_enabled():
            # BERT convention: 1 = real token, 0 = padding.
            out["attention_mask"] = inputs["attention_mask"]
        return out

    def encode_image(
        self,
        images: Union[List, List[List]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Preprocess multi-view images for SigLIP2 (256x256).

        Parameters
        ----------
        images : List or List[List]
            Single sample: [img1, img2, ...]
            Batch: [[img1a, img1b], [img2a, ...], ...]
        """
        if not isinstance(images[0], (list, tuple)):
            images = [images]

        batch_imgs, batch_masks = [], []

        for sample_imgs in images:
            processed_tensors = []

            for img in sample_imgs:
                if isinstance(img, np.ndarray):
                    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                elif isinstance(img, torch.Tensor):
                    if img.dim() == 3 and img.shape[0] != 3:
                        tensor = img.permute(2, 0, 1).float()
                    else:
                        tensor = img.float()
                    if tensor.max() > 1.0:
                        tensor = tensor / 255.0
                elif isinstance(img, Image.Image):
                    np_img = np.array(img)
                    tensor = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")

                _, h, w = tensor.shape
                if h != self.image_size or w != self.image_size:
                    tensor = F.interpolate(
                        tensor.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode='bicubic',
                        align_corners=False,
                        antialias=True,
                    ).squeeze(0)

                tensor = (tensor - self._image_mean.squeeze(0)) / self._image_std.squeeze(0)
                processed_tensors.append(tensor)

            V_exist = len(processed_tensors)
            if V_exist > 0:
                processed = torch.stack(processed_tensors, dim=0)
            else:
                processed = torch.zeros(0, 3, self.image_size, self.image_size)

            if V_exist < self.num_views:
                processed = torch.cat(
                    [processed,
                     processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                    dim=0,
                )

            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            image_mask[:V_exist] = True

            batch_imgs.append(processed)
            batch_masks.append(image_mask)

        image_input = torch.stack(batch_imgs, dim=0)
        image_mask = torch.stack(batch_masks, dim=0)

        return {"image_input": image_input, "image_mask": image_mask}

    def __call__(
        self,
        images: Optional[Union[List, List[List]]] = None,
        language_instruction: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, Any] = {}

        if language_instruction is not None:
            outputs.update(self.encode_language(language_instruction))

        if images is not None:
            outputs.update(self.encode_image(images, **kwargs))

        if "input_ids" in outputs and "image_input" in outputs:
            assert outputs["input_ids"].size(0) == outputs["image_input"].size(0), (
                f"Batch mismatch: text batch {outputs['input_ids'].size(0)} "
                f"!= image batch {outputs['image_input'].size(0)}"
            )
        return outputs
