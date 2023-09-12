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

import open_clip
import glob
import os
import PIL.Image
import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser
from open_clip.pretrained import _PRETRAINED

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("text_prompts_file", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    args = parser.parse_args()

    with open(args.text_prompts_file, 'r') as f:
        text_prompts = f.readlines()
        text_prompts = [tp.strip() for tp in text_prompts]

    print(f"Found the following {len(text_prompts)} text prompts in {args.text_prompts_file}")
    print(text_prompts)

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, 
        pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)

    with torch.no_grad():
        text = tokenizer(text_prompts)
        text_embeddings = model.encode_text(text)
        text_embeddings = text_embeddings.detach().cpu().numpy()

        print(f"Saving text embeddings to {args.output_path}")
        np.save(args.output_path, text_embeddings)
    
