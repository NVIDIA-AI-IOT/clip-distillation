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
import cv2
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
    parser.add_argument("text_embeddings_path", type=str)
    parser.add_argument("--camera_device", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dim", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--text_prompts", type=str, default=None)
    parser.add_argument("--use_asp", action="store_true")
    parser.add_argument("--use_qat", action="store_true")
    args = parser.parse_args()

    if args.text_prompts is not None:

        with open(args.text_prompts, 'r') as f:
            text_prompts = f.readlines()
            text_prompts = [tp.strip() for tp in text_prompts]

        print(f"Found the following {len(text_prompts)} text prompts in {args.text_prompts}")
        print(text_prompts)
    else:
        text_prompts = None
    if args.use_qat:
        from pytorch_quantization import quant_modules
        # use QAT monkey-patching
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

    transform = Compose([
        Resize(args.image_size),
        CenterCrop(args.image_size),
        ToTensor(),
        Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])


    text_embeddings = torch.from_numpy(
        np.load(args.text_embeddings_path)
    ).to(args.device).float()

    cap = cv2.VideoCapture(args.camera_device)

    while True:

        re, img = cap.read()
        if not re:
            print("Failed to read camera.")
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(img)

        with torch.no_grad():
            image_data = transform(image).to(args.device)
            output_embedding = model(image_data[None, ...])
            probs = embedding_to_probs(
                output_embedding,
                text_embeddings
            )
            probs = probs.detach().cpu().numpy()
            probs =probs.flatten()
            prob_indices = np.argsort(probs)[::-1] # descending
        
        pid = prob_indices[0]
        label_str = f"Index {pid} ({100 * round(probs[pid], 3)}%): \"{text_prompts[pid]}\""

        print(label_str)

cap.release()