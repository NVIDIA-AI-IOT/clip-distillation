# CLIP Distillation - STL10 Experiments

This folder contains code for experimenting with knowledge distillation
of OpenCLIP on the STL10 dataset. 

## OpenCLIP Accuracy on STL10

| Name | Method | Accuracy |
|---|--------|----------|
| openclip_text | Text prompts (zero-shot) | 96.68% |
| openclip_linear | Linear head | 98.57% |

## Training from scratch vs. distilling OpenCLIP

| Experiment Name        | Student Model | Teacher Model | Data Used   | Test Accuracy |
|------------------------|---------------|---------------|-------------|---------------|
| resnet18_from_scratch  | resnet18      | None          | stl10_train | 57.93         |
| resnet18_text_train    | resnet18      | openclip_text | stl10_train | 44.04         |
| resnet18_linear_train  | resnet18      | openclip_linear | stl10_train | 60.65       |

# Training with additional data

| Experiment Name        | Student Model | Teacher Model | Data Used   | Test Accuracy |
|------------------------|---------------|---------------|-------------|---------------|
| resnet18_text_train_unlabeled    | resnet18      | openclip_text | stl10_train + stl10_unlabeled | 94.32         |
| resnet18_linear_train_unlabeled  | resnet18      | openclip_linear | stl10_train + stl10_unlabeled | 96.88 |

# Varying the student model architecture

| Experiment Name        | Student Model | Teacher Model | Data Used   | Test Accuracy |
|------------------------|---------------|---------------|-------------|---------------|
| resnet18_linear_train_unlabeled    | resnet18      | openclip_linear | stl10_train + stl10_unlabeled | 96.88         |
| resnet34_linear_train_unlabeled  | resnet34      | openclip_linear | stl10_train + stl10_unlabeled | 96.69 |
| resnet50_linear_train_unlabeled  | resnet50      | openclip_linear | stl10_train + stl10_unlabeled | 96.76 |

# Varying the distillation method

| Experiment Name        | Student Model | Teacher Model | Data Used   | Test Accuracy |
|------------------------|---------------|---------------|-------------|---------------|
| resnet18_embedding_text_train_unlabeled    | resnet18      | openclip_embedding | stl10_train + stl10_unlabeled | 94.575        |
| resnet18_embedding_linear_train_unlabeled  | resnet34      | openclip_embedding | stl10_train + stl10_unlabeled | 96.912 |


# Performance benchmarks

| Model | Image Size | Batch Size | Precision | Throughput (FPS) | Latency (ms) | Memory (MB) |
|-----------------|------------|------------|--------|-----|---------|----|
| openclip_vitb32 | 224x224 | 8 | FP16 | 335.816 | 23.82 | 1087 |
| resnet18 | 224x224 | 8 | FP16 | 1420.2  | 5.97 | 315 |