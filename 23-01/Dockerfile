FROM nvcr.io/nvidia/pytorch:23.01-py3

RUN pip install timm open_clip_torch

# upgrade pillow to fix "UnidentifiedImageError"
RUN pip install pillow --upgrade 