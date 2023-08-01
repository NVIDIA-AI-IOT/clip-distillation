# CLIP Knowledge Distillation


<img src="assets/overview.png" height="320"/>

This repository contains code and instructions that enable you
to create your own customized image classification models
with zero-labeled data, by performing knowledge distillation of OpenCLIP models.

Even if you don't need an image classifier directly, you may find this project helpful as inspiration for how you can use knowledge distillation to optimized models for inference, or as an example of how to train models with quantization aware training and structured sparsity for inference on NVIDIA Jetson. 

This project includes,

1. Scripts to search and download relevant data from the LAION database to use for distillation
2. Scripts to distil any OpenCLIP model to any Pytorch image models (timm) CNN model.
    - Supports Quantization Aware Training (QAT) for downstream INT8 inference
    - Supports training to enforce 2:4 structured sparsity with the ASP library
3. Scripts to run inference with NVIDIA TensorRT
    - Supports INT8 model
    - Supports acceleration of 2:4 structured sparse models on certain NVIDIA Jetson platforms, like NVIDIA Jetson Orin Nano.

To get started, follow the instructions below.

> If you're new to the subject, check out our tutorial [jetson-intro-to-distillation](https://github.com/NVIDIA-AI-IOT/jetson-intro-to-distillation)
> for an introduction to knowledge distillation!


## Instructions

1. [Step 1 - Search and download relevant unlabeled images to use for distillation](#step-1)
2. [Step 2 - Pre-compute OpenCLIP embeddings](#step-2)
3. [Step 3 - Train the student CNN model to mimic the OpenCLIP model](#step-3)
4. [Step 4 - Run inference using the distilled model](#step-4)
5. [Step 5 (advanced) - Train a student model with structured sparsity](#step-5)
6. [Step 6 (advanced) - Train a student with Quantization aware training and INT8 precision](#step-6)
7. [Next Steps](#next-steps)

<a name="step-1"></a>

## Step 1 - Search and download images with CLIP filtering

### Search for relevant image URLs in the LAION database using CLIP filtering

First, let's query for image URLS that match a set of text prompts by using
the LAION clip retrieval service.

First, create a file ``data/text_prompts.txt`` with the text prompts to query.
Each prompt should exist on it's own line.

```txt
a dog
a cat
```

Next, query the images based on the text prompts.

```bash
python3 search_clip_images.py \
    "data/text_prompts.txt" \
    "data/image_urls.txt" \
    -n 5000 \
    -m 10000 \
    --max_workers 2 \
    --append
```

> Note: We use our own script rather than clip_retrieval because it does not expose some parameters and limits the number of images downloaded to a larger extent.

For the full set of arguments please type

```bash
python3 search_clip_images.py --help
```

### Download images from URL file

Next, we call the following script to download images to an output folder.
Images are assigned a unique ID based on their URL using the uuid library.  
This allows us to track the association of images with their URLs without
needing extra metadata or files.

```bash
python3 download_images.py \
    "data/image_urls.txt" \
    "data/images" \
    --max_workers 32 \
    --timeout 2
```

For the full set of arguments please type

```bash
python3 download_images.py --help
```

<a name="step-2"></a>

## Step 2 - Compute OpenCLIP embeddings

Now that we've downloaded a set of images, let's compute OpenCLIP embeddings.
This will speed up training, so we don't have to run CLIP in the training loop.

```bash
python3 compute_openclip_embeddings.py \
    data/images \
    data/embeddings \
    --batch_size 16 \
    --num_workers 8 \
    --model_name ViT-B-32 \
    --pretrained laion2b_s34b_b79k
```

> Note: For available model names and pretrained weight identifiers please reference [OpenCLIP Repo](https://github.com/mlfoundations/open_clip/blob/fb72f4db1b17133befd6c67c9cf32a533b85a321/src/open_clip/pretrained.py#L227).

For the full set of arguments please type

```bash
python3 compute_openclip_embeddings.py --help
```


<a name="step-3"></a>

## Step 3 - Train the student CNN model to mimic the OpenCLIP model

```bash
python3 distil_model_embeddings.py \
    resnet18 \
    data/images \
    data/embeddings \
    data/models/resnet18 \
    --output_dim 512 \
    --pretrained
```

For the full set of arguments please type

```bash
python3 distil_model_embeddings.py --help
```


<a name="step-4"></a>

## Step 4 - Run inference using the distilled model

### Compute text embeddings

Before we can use our distilled model for classification, we need to compute 
the text embeddings.

```bash
python3 compute_openclip_text_embeddings.py \
    data/text_prompts.txt \
    data/text_embeddings.npy \
    --model_name ViT-B-32
```

The model name should match that in step (3).  The text prompts here match
those we used for search.  If you distilled the model with other text prompts,
you could set this to just the prompts you want to use for classification.

### Predict single image with PyTorch

To run inference on a single image with PyTorch

```bash
python3 predict_pytorch.py \
    resnet18 \
    data/models/resnet18/checkpoint.pth \
    data/text_embeddings.npy \
    assets/cat.jpg \
    --text_prompts data/text_prompts.txt
```


### Live demo with camera

To run inference on a live camera feed and print results to terminal

```bash
python3 demo_pytorch.py \
    resnet18 \
    data/models/resnet18/checkpoint.pth \
    data/text_embeddings.npy \
    --text_prompts data/text_prompts.txt \
    --camera_device 0
```

## Step 5 (advanced) - Train a student model with structured sparsity

### Distil

```bash
python3 distil_model_embeddings.py \
    resnet18 \
    data/images \
    data/embeddings \
    data/models/resnet18_sparse \
    --output_dim 512 \
    --pretrained \
    --init_checkpoint data/models/resnet18/checkpoint.pth \
    --use_asp \
    --num_epochs 25
```

### Predict with PyTorch

```bash
python3 predict_pytorch.py \
    resnet18 \
    data/models/resnet18_sparse/checkpoint.pth \
    data/text_embeddings.npy \
    assets/cat.jpg \
    --text_prompts data/text_prompts.txt \
    --use_asp
```

### Demo with PyTorch

```bash
python3 demo_pytorch.py \
    resnet18 \
    data/models/resnet18_sparse/checkpoint.pth \
    data/text_embeddings.npy \
    --text_prompts data/text_prompts.txt \
    --camera_device 0 \
    --use_asp
```

### Export to ONNX

```bash
python3 export_onnx.py \
    resnet18 \
    data/models/resnet18_sparse/checkpoint.pth \
    data/onnx/resnet18_sparse.onnx \
    --use_asp
```


<a name="step-6"></a>

## Step 6 (advanced) - Train a student with Quantization aware training and INT8 precision

### Distil

```bash
python3 distil_model_embeddings.py \
    resnet18 \
    data/images \
    data/embeddings \
    data/models/resnet18_qat \
    --output_dim 512 \
    --pretrained \
    --init_checkpoint data/models/resnet18/checkpoint.pth \
    --use_qat \
    --num_epochs 25
```

### Predict with PyTorch

```bash
python3 predict_pytorch.py \
    resnet18 \
    data/models/resnet18_sparse/checkpoint.pth \
    data/text_embeddings.npy \
    assets/cat.jpg \
    --text_prompts data/text_prompts.txt \
    --use_qat
```

### Demo with PyTorch

```bash
python3 demo_pytorch.py \
    resnet18 \
    data/models/resnet18_sparse/checkpoint.pth \
    data/text_embeddings.npy \
    --text_prompts data/text_prompts.txt \
    --camera_device 0 \
    --use_qat
```

### Export to ONNX

```bash
python3 export_onnx.py \
    resnet18 \
    data/models/resnet18_qat/checkpoint.pth \
    data/onnx/resnet18_qat.onnx \
    --use_qat
```

<a name="next-steps"></a>

## Next steps

We hope you found this project helpful and that you were able to train your own image classification model, without using any labeled data.  

As a next step, we recommend reading through the source code to see how we used knoweldge distillation in this project.  We also recommend reading the source code to see how you can train a model with the convenient libraries in PyTorch for quantization aware training and structured sparsity, for more optimized inference on Jetson.

If you have any questions, or run into any issues, please let us know by opening an issue on GitHub!