#!/usr/bin/env python3
"""Single-dataset SurgVLP fine-tuning with VeRA on FC projection layer only.

Workflow:
1) Load SurgVLP base weights (SurgVLP.pth).
2) Freeze everything.
3) Inject VeRA on image FC projection layer (`backbone_img.global_embedder`).
4) Train only VeRA lambda parameters on one dataset split.
5) Merge VeRA delta into FC weights and save a full model checkpoint (e.g., SurgVLP_d1.pth).
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from mmengine.config import Config
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

# Avoid HF tokenizer fork warnings when DataLoader uses worker processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure local package import works even when launched from outside repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import surgvlp  # noqa: E402


class VeRALinear(nn.Module):
    """VeRA wrapper for Linear layers."""

    def __init__(self, base_layer: nn.Linear, rank: int, dropout: float, d_initial: float, seed: int):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"VeRALinear expects nn.Linear, got {type(base_layer)}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        vera_A = torch.randn(rank, in_features, generator=generator, device=device, dtype=dtype)
        vera_B = torch.randn(out_features, rank, generator=generator, device=device, dtype=dtype)
        vera_A = vera_A / max(in_features, 1) ** 0.5
        vera_B = vera_B / max(rank, 1) ** 0.5

        self.register_buffer("vera_A", vera_A, persistent=True)
        self.register_buffer("vera_B", vera_B, persistent=True)
        self.vera_lambda_d = nn.Parameter(torch.full((rank,), float(d_initial), device=device, dtype=dtype))
        self.vera_lambda_b = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        h = F.linear(self.dropout(x), self.vera_A)
        h = h * self.vera_lambda_d
        delta = F.linear(h, self.vera_B)
        delta = delta * self.vera_lambda_b
        return base + delta


class KvasirFolderDataset(Dataset):
    """Folder-per-class Kvasir dataset with fixed class ordering."""

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root: str, class_order: List[str], transform=None):
        self.root = root
        self.transform = transform
        self.class_order = class_order
        self.class_to_idx = {c: i for i, c in enumerate(self.class_order)}

        self.samples: List[str] = []
        self.labels: List[int] = []

        for class_name in self.class_order:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                raise RuntimeError(f"Kvasir class directory missing: {class_dir}")
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(self.IMAGE_EXTS):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[class_name])

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        label = self.labels[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"video": img, "label": label}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-dataset SurgVLP training with VeRA FC-only")
    parser.add_argument("--dataset", type=str, required=True, choices=["kvasir", "temset", "cholect50"])
    parser.add_argument("--split-tag", type=str, default="d1", choices=["d1", "d2", "d3"])

    parser.add_argument("--config", type=str, default="/DATA3/SurgVLP-main/tests/config_surgvlp.py")
    parser.add_argument("--pretrain", type=str, default="/DATA3/.cache/surgvlp/SurgVLP.pth")
    parser.add_argument("--output-dir", type=str, default="/DATA3/SurgVLP-main/outputs/single_dataset_vera")
    parser.add_argument(
        "--save-model",
        type=str,
        default="",
        help="Output full checkpoint path. Default: <output-dir>/SurgVLP_<split-tag>.pth",
    )
    parser.add_argument(
        "--save-best-acc-model",
        type=str,
        default="",
        help="Output full checkpoint path for best eval_acc. Default: <output-dir>/SurgVLP_<split-tag>_best_eval_acc.pth",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default="",
        help="CSV path for per-epoch logs. Default: <output-dir>/train_log_<dataset>_<split-tag>.csv",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--logit-scale", type=float, default=20.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--best-metric",
        type=str,
        default="eval_loss",
        choices=["eval_loss", "eval_acc", "train_loss", "train_acc"],
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop after N non-improving epochs on best-metric (<=0 disables).",
    )

    parser.add_argument("--vera-rank", type=int, default=256)
    parser.add_argument("--vera-dropout", type=float, default=0.05)
    parser.add_argument("--vera-d-initial", type=float, default=0.1)
    parser.add_argument("--img-proj-module", type=str, default="backbone_img.global_embedder")

    # Optional text projection adaptation.
    parser.add_argument("--use-text-proj", action="store_true")
    parser.add_argument("--text-proj-module", type=str, default="backbone_text.model.pooler.dense")
    parser.add_argument("--strict-text-proj", action="store_true")

    # Dataset roots.
    parser.add_argument("--kvasir-root", type=str, default="/DATA3/TDA/kvasir-dataset-v2")
    parser.add_argument("--temset-video-root", type=str, default="/DATA3/temset/microclip-frames")
    parser.add_argument("--cholec-video-root", type=str, default="/DATA3/evr-main/CholecT50/data")

    # Optional custom split source override.
    parser.add_argument("--kvasir-split-json", type=str, default="")
    parser.add_argument("--temset-train-ann", type=str, default="")
    parser.add_argument("--temset-val-ann", type=str, default="")
    parser.add_argument("--cholect50-train-csv-root", type=str, default="")
    parser.add_argument("--cholect50-val-csv-root", type=str, default="")
    parser.add_argument("--class-prompt", type=str, default="")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.6, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def default_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_module(root: nn.Module, dotted_name: str) -> nn.Module:
    module = root
    for part in dotted_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def get_parent_and_child(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def set_child_module(parent: nn.Module, child: str, new_module: nn.Module) -> None:
    if child.isdigit():
        parent[int(child)] = new_module
    else:
        setattr(parent, child, new_module)


def freeze_all_weights(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def freeze_batchnorm(model: nn.Module) -> None:
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for module in model.modules():
        if isinstance(module, bn_types):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def inject_vera_on_module(
    model: nn.Module,
    module_name: str,
    rank: int,
    dropout: float,
    d_initial: float,
    seed: int,
) -> bool:
    try:
        target = get_module(model, module_name)
    except AttributeError:
        return False
    if not isinstance(target, nn.Linear):
        return False
    parent, child = get_parent_and_child(model, module_name)
    wrapper = VeRALinear(target, rank=rank, dropout=dropout, d_initial=d_initial, seed=seed)
    set_child_module(parent, child, wrapper)
    return True


def collect_labels(dataset: Dataset) -> List[int]:
    if isinstance(dataset, Subset):
        base = collect_labels(dataset.dataset)
        return [base[i] for i in dataset.indices]

    if isinstance(dataset, ConcatDataset):
        labels = []
        for d in dataset.datasets:
            labels.extend(collect_labels(d))
        return labels

    if hasattr(dataset, "label_list"):
        raw = [int(x) for x in dataset.label_list]
        if raw and min(raw) >= 1:
            return [x - 1 for x in raw]
        return raw

    if hasattr(dataset, "labels"):
        return [int(x) for x in dataset.labels]

    labels = []
    for i in range(len(dataset)):
        labels.append(int(dataset[i]["label"]))
    return labels


def read_prompts(path: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                texts.append(t)
    if not texts:
        raise RuntimeError(f"No prompts found in {path}")
    return texts


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def assert_only_vera_trainable(model: nn.Module) -> None:
    bad = [n for n, p in model.named_parameters() if p.requires_grad and ".vera_" not in n]
    if bad:
        preview = ", ".join(bad[:8])
        tail = " ..." if len(bad) > 8 else ""
        raise RuntimeError(
            "Non-VeRA parameters are trainable, which violates frozen-backbone training: "
            f"{preview}{tail}"
        )


def infer_temset_val_ann(train_ann: str) -> str:
    if train_ann.endswith("_train.txt"):
        return train_ann[: -len("_train.txt")] + "_val.txt"
    if train_ann.endswith(".txt"):
        return train_ann[: -len(".txt")] + "_val.txt"
    return train_ann + "_val.txt"


def infer_cholect50_val_csv_root(train_csv_root: str) -> str:
    normalized = train_csv_root.rstrip("/")
    if normalized.endswith("train_csvs"):
        return normalized[: -len("train_csvs")] + "val_csvs"
    return os.path.join(os.path.dirname(normalized), "val_csvs")


def resolve_split_defaults(args: argparse.Namespace) -> Tuple[str, str, str]:
    tag_map = {
        "d1": "train4pct_val1pct_of_remaining_seed42",
        "d2": "train8pct_val2pct_of_remaining_seed42",
        "d3": "train16pct_val3pct_of_remaining_seed42",
    }
    suffix = tag_map[args.split_tag]

    if args.dataset == "kvasir":
        split_json = args.kvasir_split_json or (
            f"/DATA3/SurgVLP-main/outputs/splits/kvasir_percent_splits/"
            f"kvasir_{suffix}.json"
        )
        prompt = args.class_prompt or "/DATA3/SurgVLP-main/tests/class_prompt_kvasir.txt"
        return split_json, split_json, prompt

    if args.dataset == "temset":
        train_ann = args.temset_train_ann or (
            f"/DATA3/SurgVLP-main/outputs/splits/temset_percent_splits/"
            f"temset_{suffix}_train.txt"
        )
        val_ann = args.temset_val_ann or infer_temset_val_ann(train_ann)
        prompt = args.class_prompt or "/DATA3/SurgVLP-main/tests/class_prompt_temset.txt"
        return train_ann, val_ann, prompt

    # cholect50
    train_csv_root = args.cholect50_train_csv_root or (
        f"/DATA3/SurgVLP-main/outputs/splits/cholect50_percent_splits/"
        f"cholect50_{suffix}/train_csvs"
    )
    val_csv_root = args.cholect50_val_csv_root or infer_cholect50_val_csv_root(train_csv_root)
    prompt = args.class_prompt or "/DATA3/SurgVLP-main/tests/class_prompt.txt"
    return train_csv_root, val_csv_root, prompt


def build_temset_dataset(ann_file: str, video_root: str, transform) -> Dataset:
    cfg = dict(
        type="Recognition_temset",
        ann_file=ann_file,
        video_root=video_root,
        frame_policy="center",
        transforms=transform,
    )
    return surgvlp.load_dataset(cfg)


def build_cholect50_dataset(csv_root: str, video_root: str, transform) -> Dataset:
    vids = sorted(
        [
            f[len("video_") : -len(".csv")]
            for f in os.listdir(csv_root)
            if f.startswith("video_") and f.endswith(".csv")
        ],
        key=lambda x: int(x.replace("VID", "")),
    )
    if not vids:
        raise RuntimeError(f"No video_*.csv files found under {csv_root}")
    cfgs = [
        dict(
            type="Recognition_frame",
            csv_root=csv_root,
            vid=f"video_{vid}.csv",
            video_root=video_root,
            transforms=transform,
        )
        for vid in vids
    ]
    datasets = [surgvlp.load_dataset(c) for c in cfgs]
    return ConcatDataset(datasets)


def build_train_val_datasets(
    args: argparse.Namespace,
    train_source: str,
    val_source: str,
    train_transform,
    eval_transform,
) -> Tuple[Dataset, Dataset]:
    if args.dataset == "kvasir":
        class_order = [
            "dyed-lifted-polyps",
            "dyed-resection-margins",
            "esophagitis",
            "normal-cecum",
            "normal-pylorus",
            "normal-z-line",
            "polyps",
            "ulcerative-colitis",
        ]
        with open(train_source, "r", encoding="utf-8") as f:
            payload = json.load(f)
        train_idx = payload["dataset"]["train_indices"]
        val_idx = payload["dataset"]["val_indices"]
        if not train_idx:
            raise RuntimeError(f"No train indices found in {train_source}")
        if not val_idx:
            raise RuntimeError(f"No val indices found in {train_source}")
        ds_train_full = KvasirFolderDataset(root=args.kvasir_root, class_order=class_order, transform=train_transform)
        ds_eval_full = KvasirFolderDataset(root=args.kvasir_root, class_order=class_order, transform=eval_transform)
        return Subset(ds_train_full, train_idx), Subset(ds_eval_full, val_idx)

    if args.dataset == "temset":
        return (
            build_temset_dataset(train_source, args.temset_video_root, train_transform),
            build_temset_dataset(val_source, args.temset_video_root, eval_transform),
        )

    return (
        build_cholect50_dataset(train_source, args.cholec_video_root, train_transform),
        build_cholect50_dataset(val_source, args.cholec_video_root, eval_transform),
    )


@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    loader: DataLoader,
    tokens: torch.Tensor,
    device: torch.device,
    logit_scale: float,
    label_smoothing: float,
) -> Tuple[float, float]:
    model.eval()
    txt = model(None, tokens, mode="text")["text_emb"]
    txt = F.normalize(txt, dim=-1)

    loss_sum = 0.0
    correct = 0
    total = 0
    for batch in loader:
        images = batch["video"].to(device, non_blocking=True)
        y = batch["label"].long().to(device, non_blocking=True)
        img = model(images, None, mode="video")["img_emb"]
        img = F.normalize(img, dim=-1)
        logits = logit_scale * (img @ txt.T)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        bsz = y.numel()
        loss_sum += loss.item() * bsz
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += bsz

    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def merge_vera_linear(wrapper: VeRALinear) -> nn.Linear:
    base = wrapper.base_layer
    merged = nn.Linear(base.in_features, base.out_features, bias=(base.bias is not None))
    merged = merged.to(device=base.weight.device, dtype=base.weight.dtype)
    with torch.no_grad():
        # W_delta = (vera_B * lambda_b[:,None]) @ (vera_A * lambda_d[:,None])
        b_scaled = wrapper.vera_B * wrapper.vera_lambda_b.view(-1, 1)
        a_scaled = wrapper.vera_A * wrapper.vera_lambda_d.view(-1, 1)
        delta_w = b_scaled @ a_scaled
        merged.weight.copy_(base.weight + delta_w)
        if base.bias is not None:
            merged.bias.copy_(base.bias)
    return merged


def merge_vera_modules_inplace(model: nn.Module, target_modules: List[str]) -> None:
    for mod_name in target_modules:
        target = get_module(model, mod_name)
        if isinstance(target, VeRALinear):
            parent, child = get_parent_and_child(model, mod_name)
            merged = merge_vera_linear(target)
            set_child_module(parent, child, merged)


def extract_vera_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if ".vera_" in k}


def load_vera_state(model: nn.Module, vera_state: Dict[str, torch.Tensor]) -> None:
    if not vera_state:
        raise RuntimeError("No VeRA parameters available to load.")
    model.load_state_dict(vera_state, strict=False)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_source, val_source, prompt_path = resolve_split_defaults(args)
    if not os.path.exists(train_source):
        raise RuntimeError(f"Train split source does not exist: {train_source}")
    if not os.path.exists(val_source):
        raise RuntimeError(f"Val split source does not exist: {val_source}")

    cfg = Config.fromfile(args.config)["config"]
    model, _ = surgvlp.load(cfg.model_config, device=device, pretrain=args.pretrain)
    model = model.to(device)

    freeze_all_weights(model)
    freeze_batchnorm(model)

    target_modules = []
    ok_img = inject_vera_on_module(
        model,
        args.img_proj_module,
        rank=args.vera_rank,
        dropout=args.vera_dropout,
        d_initial=args.vera_d_initial,
        seed=args.seed,
    )
    if not ok_img:
        raise RuntimeError(f"Image projection module not found or not Linear: {args.img_proj_module}")
    target_modules.append(args.img_proj_module)

    if args.use_text_proj:
        ok_txt = inject_vera_on_module(
            model,
            args.text_proj_module,
            rank=args.vera_rank,
            dropout=args.vera_dropout,
            d_initial=args.vera_d_initial,
            seed=args.seed + 1,
        )
        if not ok_txt:
            msg = f"Text projection module not found or not Linear: {args.text_proj_module}"
            if args.strict_text_proj:
                raise RuntimeError(msg)
            print(f"[Warn] {msg}. Continuing with image projection only.")
        else:
            target_modules.append(args.text_proj_module)

    assert_only_vera_trainable(model)

    train_dataset, val_dataset = build_train_val_datasets(
        args,
        train_source=train_source,
        val_source=val_source,
        train_transform=default_transform(),
        eval_transform=default_eval_transform(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_labels = collect_labels(train_dataset)
    val_labels = collect_labels(val_dataset)
    prompts = read_prompts(prompt_path)
    class_count = len(prompts)
    all_labels = train_labels + val_labels
    if all_labels and max(all_labels) >= class_count:
        raise RuntimeError(
            f"Label count mismatch: max_label={max(all_labels)} but prompts={class_count} ({prompt_path})"
        )
    tokens = surgvlp.tokenize(prompts, device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found.")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total, trainable = count_params(model)
    log_csv_path = args.log_csv or os.path.join(
        args.output_dir, f"train_log_{args.dataset}_{args.split_tag}.csv"
    )
    os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
    print(f"[Info] dataset={args.dataset} split_tag={args.split_tag}")
    print(f"[Info] train_source={train_source}")
    print(f"[Info] val_source={val_source}")
    print(f"[Info] prompt_file={prompt_path}")
    print(f"[Info] target_modules={target_modules}")
    print(f"[Info] trainable_params={trainable:,}/{total:,} ({100.0 * trainable / total:.6f}%)")
    print(f"[Info] train_samples={len(train_dataset)} val_samples={len(val_dataset)} classes={class_count}")
    print(f"[Info] best_metric={args.best_metric}")
    print(f"[Info] log_csv={log_csv_path}")
    if args.early_stop_patience > 0:
        print(f"[Info] early_stop_patience={args.early_stop_patience}")
    else:
        print("[Info] early_stop_patience=disabled")

    best_metric_name = args.best_metric
    best_metric_value: Optional[float] = None
    best_epoch = 0
    best_vera_state: Dict[str, torch.Tensor] = {}
    no_improve_epochs = 0

    csv_fields = [
        "epoch",
        "train_loss",
        "train_acc",
        "eval_loss",
        "eval_acc",
        "best_metric_name",
        "metric_value",
        "improved",
        "best_metric_value",
        "best_epoch",
        "no_improve_epochs",
    ]
    with open(log_csv_path, "w", newline="", encoding="utf-8") as log_file:
        csv_writer = csv.DictWriter(log_file, fieldnames=csv_fields)
        csv_writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            freeze_batchnorm(model)

            loss_sum = 0.0
            correct = 0
            total_n = 0

            for step, batch in enumerate(train_loader, start=1):
                images = batch["video"].to(device, non_blocking=True)
                y = batch["label"].long().to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                img = model(images, None, mode="video")["img_emb"]
                txt = model(None, tokens, mode="text")["text_emb"]
                img = F.normalize(img, dim=-1)
                txt = F.normalize(txt, dim=-1)
                logits = args.logit_scale * (img @ txt.T)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
                loss.backward()

                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()

                bsz = y.numel()
                total_n += bsz
                loss_sum += loss.item() * bsz
                correct += (logits.argmax(dim=1) == y).sum().item()

                if step % 20 == 0 or step == len(train_loader):
                    print(
                        f"[Epoch {epoch}/{args.epochs}] step={step}/{len(train_loader)} "
                        f"loss={loss_sum / max(total_n, 1):.4f} acc={correct / max(total_n, 1):.4f}"
                    )

            train_loss = loss_sum / max(total_n, 1)
            train_acc = correct / max(total_n, 1)
            eval_loss, eval_acc = evaluate_task(
                model,
                val_loader,
                tokens,
                device,
                args.logit_scale,
                args.label_smoothing,
            )

            metric_map = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
            }
            metric_value = metric_map[best_metric_name]
            improved = False
            if best_metric_value is None:
                improved = True
            elif best_metric_name in ("train_acc", "eval_acc"):
                improved = metric_value > best_metric_value
            else:
                improved = metric_value < best_metric_value

            best_msg = ""
            if improved:
                best_metric_value = metric_value
                best_epoch = epoch
                best_vera_state = extract_vera_state(model)
                no_improve_epochs = 0
                best_msg = " best=updated"
            else:
                no_improve_epochs += 1

            csv_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.8f}",
                    "train_acc": f"{train_acc:.8f}",
                    "eval_loss": f"{eval_loss:.8f}",
                    "eval_acc": f"{eval_acc:.8f}",
                    "best_metric_name": best_metric_name,
                    "metric_value": f"{metric_value:.8f}",
                    "improved": int(improved),
                    "best_metric_value": "" if best_metric_value is None else f"{best_metric_value:.8f}",
                    "best_epoch": best_epoch,
                    "no_improve_epochs": no_improve_epochs,
                }
            )
            log_file.flush()

            print(
                f"[Epoch {epoch}/{args.epochs}] done "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} "
                f"saved_metric({best_metric_name})={metric_value:.4f}{best_msg}"
            )

            if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                print(
                    f"[EarlyStop] No improvement in {best_metric_name} for {no_improve_epochs} epoch(s). "
                    f"Stopping at epoch {epoch}."
                )
                break

    print(f"[Info] wrote csv log: {log_csv_path}")

    if best_vera_state:
        load_vera_state(model, best_vera_state)
        print(f"[Info] loaded best VeRA state from epoch {best_epoch} ({best_metric_name}={best_metric_value:.4f})")
    else:
        print("[Warn] Best VeRA state was not tracked; saving final epoch state.")

    # Merge VeRA modules into base FC module(s) and save a full plain state_dict.
    model.eval()
    merge_vera_modules_inplace(model, target_modules)

    if args.save_model:
        save_path = args.save_model
    else:
        save_path = os.path.join(args.output_dir, f"SurgVLP_{args.split_tag}.pth")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[Done] wrote merged full model checkpoint: {save_path}")


if __name__ == "__main__":
    main()
