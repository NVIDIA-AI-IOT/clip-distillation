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
    parser.add_argument("output_path", type=str)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()


    model_clip, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, 
        pretrained=args.pretrained
    )

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model.encode_image(x)

    model = ModelWrapper(model_clip)
    model = model.cuda().eval()

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