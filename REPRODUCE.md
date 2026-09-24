# REPRODUCE.md 


## 0. Setup

```bash
git clone https://github.com/DarakshanRashid/RobustEndoCLIP.git
cd RobustEndoCLIP
pip install -r requirements.txt
pip install -e .
```

Base pretrained models (SurgVLP, HecVL, PeskaVLP): download the weights from the
[CAMMA SurgVLP repository](https://github.com/CAMMA-public/SurgVLP) and save them as
`checkpoints/SurgVLP.pth`, `checkpoints/HecVL.pth`, `checkpoints/PeskaVLP.pth`.

## 1. Data

You will need, per dataset, both a **clean** root and an **Endo-C6 corrupted**
root (six subdirectories, named exactly as below since `scripts/infer.sh` looks for these names:
`defocus_blur_s5_p100`, `fog_s5_p100`, `shot_noise_s5_p100`,
`motion_blur_p100_s5`, `packet_loss_p100_i5`, `smoke_p100_s5`).


**Endo-C6 generation code is not included in this release** 
only the pre-generated corrupted data is linked above. If you need to
regenerate corruptions at other severities, that code lives outside this
repository; ask the authors.


## 2. Generate the 4/8/16% label-budget splits

```bash
#CholecT50
python scripts/make_cholect50_percent_splits.py \
  --csv-root /path/to/cholect50/csvs \
  --output-dir ./splits/cholect50_percent_splits \
  --seed 42

# Kvasir
python scripts/make_kvasir_percent_splits.py \
  --root /path/to/kvasir-dataset-v2 \
  --output-dir ./splits/kvasir_percent_splits \
  --seed 42

# TEMSET-24K
python scripts/make_temset_percent_splits.py \
  --train-ann /path/to/temset_splits/train.txt \
  --output-dir ./splits/temset_percent_splits \
  --seed 42
```


## 3. SurgVLP / HecVL / PeskaVLP -- inference only (no training)


```bash
CHECKPOINT=./checkpoints/SurgVLP.pth   CONFIG=tests/config_surgvlp.py   ... bash scripts/infer.sh
CHECKPOINT=./checkpoints/HecVL.pth     CONFIG=tests/config_hecvl.py     ... bash scripts/infer.sh
CHECKPOINT=./checkpoints/PeskaVLP.pth  CONFIG=tests/config_peskavlp.py  ... bash scripts/infer.sh
```

## 4. RobustEndoCLIP -training


### 4a. RobustEndoCLIP-VeRA, 3 chained stages per budget

```bash
SPLIT_TAG=d3 \
BASE_CHECKPOINT=./checkpoints/SurgVLP.pth \
KVASIR_ROOT=/path/to/kvasir-dataset-v2 \
CHOLECT50_ROOT=/path/to/cholect50/data \
CHOLECT50_TRAIN_CSV_ROOT=./splits/cholect50_percent_splits/cholect50_train16pct_val3pct_of_remaining_seed42/train_csvs \
CHOLECT50_VAL_CSV_ROOT=./splits/cholect50_percent_splits/cholect50_train16pct_val3pct_of_remaining_seed42/val_csvs \
TEMSET_ROOT=/path/to/temset/microclip-frames \
TEMSET_TRAIN_ANN=./splits/temset_percent_splits/temset_train16pct_val3pct_of_remaining_seed42_train.txt \
TEMSET_VAL_ANN=./splits/temset_percent_splits/temset_train16pct_val3pct_of_remaining_seed42_val.txt \
  bash scripts/train_robustendoclip_vera.sh
```

`SPLIT_TAG` = `d1` (4%), `d2` (8%), or `d3` (16%) -- the script maps this to
the checkpoint's filename automatically. Final output:
`checkpoints/RobustEndoCLIP_VeRA_4pct.pth` / `_8pct.pth` / `_16pct.pth`
(matching `SPLIT_TAG`) 

### 4b. RobustEndoCLIP-LoRA 


```bash
BASE_CHECKPOINT=./checkpoints/SurgVLP.pth \
KVASIR_ROOT=/path/to/kvasir-dataset-v2 \
CHOLECT50_ROOT=/path/to/cholect50/data \
CHOLECT50_TRAIN_CSV_ROOT=./splits/cholect50_percent_splits/cholect50_train16pct_val3pct_of_remaining_seed42/train_csvs \
CHOLECT50_VAL_CSV_ROOT=./splits/cholect50_percent_splits/cholect50_train16pct_val3pct_of_remaining_seed42/val_csvs \
TEMSET_ROOT=/path/to/temset/microclip-frames \
TEMSET_TRAIN_ANN=./splits/temset_percent_splits/temset_train16pct_val3pct_of_remaining_seed42_train.txt \
TEMSET_VAL_ANN=./splits/temset_percent_splits/temset_train16pct_val3pct_of_remaining_seed42_val.txt \
  bash scripts/train_robustendoclip_lora.sh
```

Final output: `checkpoints/RobustEndoCLIP_LoRA_16pct.pth`.

## 5. Evaluation (clean + Endo-C6, all three datasets, any checkpoint)

```bash
CHECKPOINT=./checkpoints/RobustEndoCLIP_VeRA_16pct.pth \
CONFIG=tests/config_surgvlp.py \
CHOLECT50_ROOT=/path/to/cholect50/data \
CHOLECT50_CSV_ROOT=/path/to/cholect50/csvs \
CHOLECT50_CORR_ROOT=/path/to/cholect50_endoc6_corruptions \
KVASIR_ROOT=/path/to/kvasir_clean_test \
KVASIR_CORR_ROOT=/path/to/kvasir_endoc6_corruptions \
TEMSET_ROOT=/path/to/temset_eval_frames \
TEMSET_ANN_ROOT=/path/to/temset_endoc6_ann \
TEMSET_CORR_ROOT=/path/to/temset_endoc6_corruptions \
  bash scripts/infer.sh --output-dir ./eval_outputs/robustendoclip_vera_d3
```
