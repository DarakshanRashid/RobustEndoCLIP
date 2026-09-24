# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
import os
import random
import torch
import torch.utils.data
import numpy as np
import json
import pickle as pkl
import re
from PIL import Image, ImageFile
import math
import copy
import pandas as pd
from . import utils
from ..registry import DATASETS
from torch.utils.data import Dataset

def pil_loader(path):
    # Some generated corruption frames can be truncated; allow PIL to decode
    # what is available instead of raising hard errors immediately.
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    except Exception as exc:
        raise OSError(f'Failed to load image: {path}') from exc

@DATASETS.register_module(name='Recognition_frame')
class CholecDataset(Dataset):
    FRAME_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    def __init__(self, csv_root, vid, video_root, transforms=None, loader=pil_loader):
        csv_name = os.path.join(csv_root, vid)
        df = pd.read_csv(csv_name)
        self.video_root = video_root

        self.file_list = df['path'].tolist()
        self.label_list = df['label'].tolist()
        assert len(self.file_list) == len(self.label_list)
        self.transform = transforms
        self.loader = loader

        # Detect the on-disk frame layout once to avoid per-sample probing.
        # Supported layouts:
        # 1) Flat files: <video_root>/VID06_1.png
        # 2) Nested folders: <video_root>/VID06/000000.png (zero padded)
        #    or <video_root>/VID06/0.png (no padding)
        self._path_mode = 'unknown'
        self._nested_pad = 0
        self._nested_offset = 0
        if self.file_list:
            sample = self.file_list[0]
            v_id, f_id = sample.split('.png')[0].split('_')
            frame_idx = int(f_id)
            flat_candidate = os.path.join(self.video_root, f"{v_id}_{frame_idx + 1}.png")
            if os.path.exists(flat_candidate):
                self._path_mode = 'flat'
            else:
                video_dir = os.path.join(self.video_root, v_id)
                if os.path.isdir(video_dir):
                    frame_files = []
                    numeric_indices = []
                    max_pad = 0
                    for name in os.listdir(video_dir):
                        stem, ext = os.path.splitext(name)
                        if ext.lower() not in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'):
                            continue
                        if stem.isdigit():
                            numeric_indices.append(int(stem))
                            if len(stem) > max_pad:
                                max_pad = len(stem)
                        frame_files.append(name)
                    if numeric_indices:
                        numeric_indices.sort()
                        self._path_mode = 'nested'
                        self._nested_pad = max_pad
                        available = set(numeric_indices)
                        best_offset = 0
                        best_count = -1
                        for offset in (0, 1):
                            count = 0
                            for path in self.file_list:
                                _, f_id = path.split('.png')[0].split('_')
                                idx = int(f_id) + offset
                                if idx in available:
                                    count += 1
                            if count > best_count:
                                best_count = count
                                best_offset = offset
                        self._nested_offset = best_offset

                        kept_files = []
                        kept_labels = []
                        for path, label in zip(self.file_list, self.label_list):
                            _, f_id = path.split('.png')[0].split('_')
                            idx = int(f_id) + self._nested_offset
                            if idx in available:
                                kept_files.append(path)
                                kept_labels.append(label)
                        if kept_files:
                            self.file_list = kept_files
                            self.label_list = kept_labels
                    else:
                        self._path_mode = 'unknown'
                else:
                    nested_found = False
                    for offset in (0, 1):
                        cand = os.path.join(self.video_root, v_id, f"{frame_idx + offset:06d}.png")
                        if os.path.exists(cand):
                            self._path_mode = 'nested'
                            self._nested_pad = 6
                            self._nested_offset = offset
                            nested_found = True
                            break
                    if not nested_found:
                        for offset in (0, 1):
                            cand = os.path.join(self.video_root, v_id, f"{frame_idx + offset}.png")
                            if os.path.exists(cand):
                                self._path_mode = 'nested'
                                self._nested_pad = 0
                                self._nested_offset = offset
                                nested_found = True
                                break
                    if not nested_found:
                        self._path_mode = 'unknown'

    def _flat_frame_path(self, v_id, frame_num):
        # CSV frame ids are 0-based and on-disk flat layout uses +1.
        disk_idx = frame_num + 1
        for ext in self.FRAME_EXTS:
            cand = os.path.join(self.video_root, f"{v_id}_{disk_idx}{ext}")
            if os.path.exists(cand):
                return cand
        return os.path.join(self.video_root, f"{v_id}_{disk_idx}.png")

    def _nested_frame_path(self, v_id, frame_num):
        video_dir = os.path.join(self.video_root, v_id)
        if not os.path.isdir(video_dir):
            return None
        disk_idx = frame_num + self._nested_offset

        stem_candidates = []
        if self._nested_pad:
            stem_candidates.append(f"{disk_idx:0{self._nested_pad}d}")
        stem_candidates.extend([f"{disk_idx:06d}", str(disk_idx)])

        seen = set()
        for stem in stem_candidates:
            if stem in seen:
                continue
            seen.add(stem)
            for ext in self.FRAME_EXTS:
                cand = os.path.join(video_dir, f"{stem}{ext}")
                if os.path.exists(cand):
                    return cand
        return None

    def _resolve_img_path(self, img_names):
        v_id, f_id = img_names.split('.png')[0].split('_')
        frame_idx = int(f_id)

        if self._path_mode == 'flat':
            return self._flat_frame_path(v_id, frame_idx)
        if self._path_mode == 'nested':
            nested = self._nested_frame_path(v_id, frame_idx)
            if nested is not None:
                return nested
            return self._flat_frame_path(v_id, frame_idx)

        # Unknown mode: try flat then nested probing.
        flat = self._flat_frame_path(v_id, frame_idx)
        if os.path.exists(flat):
            return flat
        nested = self._nested_frame_path(v_id, frame_idx)
        if nested is not None:
            return nested
        return flat

    def _iter_fallback_paths(self, v_id, frame_idx, max_radius=120):
        # Nearby frame fallback keeps sample count stable when a single frame is corrupt.
        for radius in range(1, max_radius + 1):
            for delta in (-radius, radius):
                cand_idx = frame_idx + delta
                if cand_idx < 0:
                    continue

                if self._path_mode == 'flat':
                    cand = self._flat_frame_path(v_id, cand_idx)
                    if os.path.exists(cand):
                        yield cand
                    continue

                if self._path_mode == 'nested':
                    cand = self._nested_frame_path(v_id, cand_idx)
                    if cand is not None:
                        yield cand
                    continue

                # Unknown mode: try both.
                cand_flat = self._flat_frame_path(v_id, cand_idx)
                if os.path.exists(cand_flat):
                    yield cand_flat
                    continue
                cand_nested = self._nested_frame_path(v_id, cand_idx)
                if cand_nested is not None:
                    yield cand_nested


    def __getitem__(self, index):
        img_names = self.file_list[index]
        v_id, f_id = img_names.split('.png')[0].split('_')
        frame_idx = int(f_id)
        img_path = self._resolve_img_path(img_names)

        try:
            imgs = self.loader(img_path)
        except OSError as exc:
            imgs = None
            for fallback_path in self._iter_fallback_paths(v_id, frame_idx):
                if fallback_path == img_path:
                    continue
                try:
                    imgs = self.loader(fallback_path)
                    break
                except OSError:
                    continue
            if imgs is None:
                raise OSError(
                    f"Failed to load frame for sample '{img_names}' (primary path: {img_path})"
                ) from exc

        labels_phase = self.label_list[index]

        # print(imgs.size)
        if self.transform is not None:
            imgs = self.transform(imgs)

        final_dict = {'video':imgs, 'label': labels_phase-1} # the label id starts from 1
        return final_dict

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module(name='Recognition_temset')
class TEMSETDataset(Dataset):
    """TEMSET frame-folder dataset.

    Annotation format:
        <clip_folder_rel_or_abs_path> <label_id>
    Example:
        clip_00001 3
    """

    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, ann_file, video_root, transforms=None, loader=pil_loader, frame_policy='center'):
        self.ann_file = ann_file
        self.video_root = video_root
        self.transform = transforms
        self.loader = loader
        self.frame_policy = frame_policy
        if self.frame_policy not in {'first', 'center', 'last'}:
            raise ValueError(f"Unsupported frame_policy: {self.frame_policy}")

        self.file_list = []
        self.label_list = []
        with open(self.ann_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                clip_path, label = line.split()
                if self.video_root is not None and not os.path.isabs(clip_path):
                    clip_path = os.path.join(self.video_root, clip_path)
                self.file_list.append(clip_path)
                self.label_list.append(int(label))

        assert len(self.file_list) == len(self.label_list)

    @staticmethod
    def _frame_index(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(stem)
        except ValueError:
            try:
                return int(float(stem))
            except ValueError:
                return None

    def _list_frames(self, clip_dir):
        if not os.path.isdir(clip_dir):
            raise FileNotFoundError(f'Clip directory not found: {clip_dir}')

        candidates = []
        for name in os.listdir(clip_dir):
            path = os.path.join(clip_dir, name)
            ext = os.path.splitext(name)[1].lower()
            if os.path.isfile(path) and ext in self.IMG_EXTENSIONS:
                candidates.append(path)
        candidates = sorted(candidates)
        if not candidates:
            raise FileNotFoundError(f'No frame files found in {clip_dir}')

        # Keep one frame per numeric index to avoid duplicates like 0.jpg and 00000.jpg.
        numeric_frames = {}
        non_numeric_frames = []
        for path in candidates:
            idx = self._frame_index(path)
            if idx is None:
                non_numeric_frames.append(path)
            elif idx not in numeric_frames:
                numeric_frames[idx] = path

        frame_paths = [numeric_frames[idx] for idx in sorted(numeric_frames)]
        frame_paths.extend(non_numeric_frames)
        return frame_paths if frame_paths else candidates

    def _pick_frame(self, frame_paths):
        if self.frame_policy == 'first':
            return frame_paths[0]
        if self.frame_policy == 'last':
            return frame_paths[-1]
        return frame_paths[len(frame_paths) // 2]

    def __getitem__(self, index):
        clip_dir = self.file_list[index]
        frame_paths = self._list_frames(clip_dir)
        frame_path = self._pick_frame(frame_paths)
        img = self.loader(frame_path)

        if self.transform is not None:
            img = self.transform(img)

        return {'video': img, 'label': self.label_list[index]}

    def __len__(self):
        return len(self.file_list)
