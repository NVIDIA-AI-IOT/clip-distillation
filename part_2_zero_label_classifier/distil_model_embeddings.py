import torch
import timm
import numpy as np
import argparse
import glob
import os
from torch.utils.data import (
    Dataset,
    DataLoader
)
from PIL import Image
import tqdm
import torch.nn.functional as F
import json
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    Resize,
    CenterCrop
)
from torch.utils.tensorboard import SummaryWriter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_image_id_from_path(image_path):
    return os.path.basename(image_path).split('.')[0]


def get_embedding_path(embedding_folder, image_id):
    return os.path.join(embedding_folder, image_id + ".npy")


def find_images(folder: str):
    image_paths = glob.glob(os.path.join(args.images_folder, "*.jpg"))
    image_paths += glob.glob(os.path.join(args.images_folder, "*.png"))
    return image_paths


class ImageEmbeddingDataset(Dataset):
    def __init__(self, image_paths, embedding_paths, transform=None):
        self.image_paths = image_paths
        self.embedding_paths = embedding_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        embedding = np.load(self.embedding_paths[index])
        return image, embedding



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("images_folder", type=str)
    parser.add_argument("embeddings_folder", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dim", type=int, default=512, help="Dimension of output embedding.  Must match the embeddings generated.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--momentum", type=float, default=0.)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sdg"])
    parser.add_argument("--criterion", type=str, default="mse", choices=["mse", "l1", "huber"])
    parser.add_argument("--use_asp", action="store_true")
    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--use_qat", action="store_true")
    args = parser.parse_args()

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, "args.json")
    print(f"Running with args {args_dict}")
    print(f"Writing args to {args_path}...")
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=2)
        
    all_image_paths = find_images(args.images_folder)
    print(f"Found {len(all_image_paths)} in the folder {args.images_folder}")

    image_paths = []
    embedding_paths = []

    for image_path in all_image_paths:
        image_id = get_image_id_from_path(image_path)
        embedding_path = get_embedding_path(args.embeddings_folder, image_id)
        if os.path.exists(embedding_path):
            image_paths.append(image_path)
            embedding_paths.append(embedding_path)
    
    print(f"Found embeddings for {len(embedding_paths)} out of {len(all_image_paths)} images.")

    if args.criterion == "mse":
        criterion = F.mse_loss
    elif args.criterion == "l1":
        criterion = F.l1_loss
    elif args.criterion == "huber":
        criterion = F.huber_loss
    else:
        raise RuntimeError(f"Unsupported criterion {args.criterion}")

    if args.use_qat:
        from pytorch_quantization import quant_modules
        # use QAT monkey-patching
        print("Initializing quantization aware training (QAT)")
        quant_modules.initialize()

    model = timm.create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_classes=args.output_dim
    )
    model = model.to(args.device)

    # Setup optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )

    transform = Compose([
        Resize(args.image_size),
        CenterCrop(args.image_size),
        ToTensor(),
        Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    dataset = ImageEmbeddingDataset(
        image_paths=image_paths,
        embedding_paths=embedding_paths,
        transform=transform
    )

    data_loader = DataLoader(
        dataset=dataset,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        batch_size=args.batch_size
    )

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1 # pick up on previous epoch
    elif args.init_checkpoint is not None and os.path.exists(args.init_checkpoint):
        checkpoint = torch.load(args.init_checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = 0  # don't use start checkpoints epoch
    else:
        start_epoch = 0

    writer_path = os.path.join(args.output_dir, "log")
    writer = SummaryWriter(writer_path)

    model = model.train()


    if args.use_asp:
        from apex.contrib.sparsity import ASP
        ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d], allow_recompute_mask=False, allow_permutation=False)
        ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks()
        print(f"Pruned model for 2:4 sparse weights using ASP")

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.

        for image, embedding in tqdm.tqdm(iter(data_loader)):
            image = image.to(args.device)
            embedding = embedding.to(args.device)
            
            optimizer.zero_grad()
            output_embedding = model(image)

            loss = criterion(output_embedding, embedding)

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)

        writer.add_scalar(
            "loss",
            scalar_value=epoch_loss,
            global_step=epoch
        )
        
        print(f"EPOCH: {epoch} - LOSS: {epoch_loss}")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        torch.save(
            checkpoint,
            checkpoint_path
        )