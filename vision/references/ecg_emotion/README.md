# ECG emotion recognition with 1D CNN models

This reference trains 1D CNN baselines on DREAMER ECG signals for four-quadrant valence-arousal emotion recognition. It currently supports `mobilenet_v3_small_1d`, `efficientnet_v2_s_1d`, `resnet50_1d`, and `densenet169_1d`.

## Task

DREAMER has 23 subjects, 18 trials per subject, 2 ECG leads, and valence/arousal ratings from 1 to 5. Following the EmoNet-ECG setup, the labels are mapped with threshold `3`:

| Class | Rule |
| --- | --- |
| `LVLA` | `valence < 3` and `arousal < 3` |
| `LVHA` | `valence < 3` and `arousal >= 3` |
| `HVLA` | `valence >= 3` and `arousal < 3` |
| `HVHA` | `valence >= 3` and `arousal >= 3` |

The model consumes raw ECG tensors shaped `[batch, 2, length]`. Each trial is z-score normalized per lead and right-padded with zeros to the longest DREAMER ECG unless `--max-length` is set.

## Environment

Use the dedicated Conda environment:

```powershell
conda activate ecg-mobilenet
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Training

Run from the repository root:

```powershell
python vision/references/ecg_emotion/train_dreamer.py `
  --dreamer-mat path/to/DREAMER.mat `
  --output-dir outputs/dreamer_mobilenetv3_1d `
  --model mobilenet_v3_small_1d `
  --epochs 100 `
  --batch-size 4 `
  --amp
```

EfficientNetV2-S is larger than MobileNetV3-Small. On a 4 GB GTX 1650 Ti, start with `--batch-size 1 --amp`:

```powershell
python vision/references/ecg_emotion/train_dreamer.py `
  --dreamer-mat path/to/DREAMER.mat `
  --output-dir outputs/dreamer_efficientnetv2_s_1d `
  --model efficientnet_v2_s_1d `
  --epochs 100 `
  --batch-size 1 `
  --amp
```

ResNet50 is also much larger than MobileNetV3-Small. On a 4 GB GTX 1650 Ti, start with `--batch-size 1 --lr 1e-4`. In local smoke tests, AMP produced non-finite loss for ResNet50, so keep this run in FP32 unless you have validated a stable AMP setting:

```powershell
python vision/references/ecg_emotion/train_dreamer.py `
  --dreamer-mat path/to/DREAMER.mat `
  --output-dir outputs/dreamer_resnet50_1d `
  --model resnet50_1d `
  --epochs 100 `
  --batch-size 1 `
  --lr 1e-4
```

DenseNet-169 is a deeper DenseNet baseline adapted to 1D ECG. On a server with RTX 4090 GPUs, start with `--batch-size 8 --lr 1e-4`:

```bash
CUDA_VISIBLE_DEVICES=0 python3 vision/references/ecg_emotion/train_dreamer.py \
  --dreamer-mat data/DREAMER.mat \
  --output-dir outputs/dreamer_densenet169_seed0 \
  --model densenet169_1d \
  --epochs 100 \
  --batch-size 8 \
  --seeds 0 \
  --lr 1e-4
```

For a fair single-seed comparison against an existing MobileNet run, reuse its subject split:

```powershell
python vision/references/ecg_emotion/train_dreamer.py `
  --dreamer-mat path/to/DREAMER.mat `
  --split-json outputs/dreamer_mobilenetv3_seed0/seed_0/subject_split.json `
  --output-dir outputs/dreamer_efficientnetv2_s_seed0 `
  --model efficientnet_v2_s_1d `
  --epochs 100 `
  --batch-size 1 `
  --seeds 0 `
  --amp
```

For the same fair split with ResNet50:

```powershell
python vision/references/ecg_emotion/train_dreamer.py `
  --dreamer-mat path/to/DREAMER.mat `
  --split-json outputs/dreamer_mobilenetv3_seed0/seed_0/subject_split.json `
  --output-dir outputs/dreamer_resnet50_seed0 `
  --model resnet50_1d `
  --epochs 100 `
  --batch-size 1 `
  --seeds 0 `
  --lr 1e-4
```

For the same fair split with DenseNet-169:

```bash
CUDA_VISIBLE_DEVICES=0 python3 vision/references/ecg_emotion/train_dreamer.py \
  --dreamer-mat data/DREAMER.mat \
  --split-json outputs/dreamer_mobilenetv3_seed0/seed_0/subject_split.json \
  --output-dir outputs/dreamer_densenet169_seed0 \
  --model densenet169_1d \
  --epochs 100 \
  --batch-size 8 \
  --seeds 0 \
  --lr 1e-4
```

For a quick CPU/GPU smoke run:

```powershell
python vision/references/ecg_emotion/train_dreamer.py `
  --dreamer-mat path/to/DREAMER.mat `
  --output-dir outputs/smoke_dreamer `
  --model mobilenet_v3_small_1d `
  --epochs 1 `
  --batch-size 2 `
  --seeds 0
```

By default the script runs seeds `0 1 2 3 4`. If no `--split-json` is provided, each seed creates a deterministic subject-level train/val/test split and saves it next to the checkpoint.

## Outputs

Each seed writes:

- `seed_<n>/subject_split.json`
- `seed_<n>/best.pth`
- `seed_<n>/history.json`
- `seed_<n>/metrics.json`

The top-level output directory writes:

- `args.json`
- `summary.json`

`summary.json` contains test accuracy and macro-F1 mean/std across seeds.

## Notes

These are 1D adaptations for raw ECG. They intentionally do not modify `torchvision.models.mobilenetv3`, `torchvision.models.efficientnet`, or `torchvision.models.densenet`, which remain image classification implementations.
