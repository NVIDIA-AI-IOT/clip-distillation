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
    parser.add_argument("input_folder", type=str)
    parser.add_argument("output_folder", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    image_paths = glob.glob(os.path.join(
        args.input_folder, "*.jpg"
    ))
    image_paths += glob.glob(os.path.join(
        args.input_folder, "*.png"
    ))

    def get_image_id_from_path(image_path):
        return os.path.basename(image_path).split('.')[0]
    
    def get_embedding_path(embedding_folder, image_id):
        return os.path.join(embedding_folder, image_id + ".npy")

    old_image_paths = image_paths
    image_paths = [
        image_path for image_path in image_paths
        if not os.path.exists(
            get_embedding_path(
                args.output_folder, 
                get_image_id_from_path(image_path)
            )
        )
    ]

    num_skip = len(old_image_paths) - len(image_paths)
    if num_skip == len(old_image_paths):
        print(f"All embeddings already computed. Nothing left to do.")
        exit()
    elif num_skip > 0:
        print(f"Skipping computation of {num_skip} embeddings because they already exist.")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, 
        pretrained=args.pretrained
    )

    class ImageDataset(Dataset):
        def __init__(self, image_paths, preproc):
            self.image_paths = image_paths
            self.preproc = preproc

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, index):
            image = PIL.Image.open(self.image_paths[index])
            image = self.preproc(image)
            return index, image

    dataset = ImageDataset(image_paths, preprocess)

    data_loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )

    print(f"Computing embeddings for {len(image_paths)} images...")
    with torch.no_grad():
        for indices, images in tqdm.tqdm(iter(data_loader)):
            count = len(indices)
            embeddings = model.encode_image(images)
            for idx in range(count):
                image_path_idx = int(indices[idx])
                image_path = dataset.image_paths[image_path_idx]
                embedding_path = get_embedding_path(
                    args.output_folder,
                    get_image_id_from_path(image_path)
                )
                embedding = embeddings[idx].detach().cpu().numpy()
                np.save(embedding_path, embedding)


