# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import timm
import torch
import PIL.Image
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def embedding_to_probs(embedding, text_embedding, temp=100.):
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    logits = embedding @ text_embedding.T
    logits = F.softmax(temp * logits, dim=-1)
    return logits


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dim", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_asp", action="store_true")
    parser.add_argument("--use_qat", action="store_true")
    args = parser.parse_args()

    if args.use_qat:
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization import quant_modules
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        # use QAT monkey-patching, must be called before model is instantiated
        print("Initializing quantization aware training (QAT)")
        quant_modules.initialize()

    model = timm.create_model(
        model_name=args.model_name,
        num_classes=args.output_dim
    )

    model = model.cuda().eval()

    if args.use_asp:
        from apex.contrib.sparsity import ASP
        ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d], allow_recompute_mask=False, allow_permutation=False)
        # ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks()

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    data = torch.randn(1, 3, args.image_size, args.image_size).cuda()

    torch.onnx.export(
        model,
        (data,),
        args.output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: "height", 3: "width"},
            'output': {0: 'batch_size', 2: "height", 3: "width"}
        }
    )
