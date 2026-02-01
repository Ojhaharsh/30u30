# Data Directory - Day 11: Dilated Convolutions

This directory stores datasets and model checkpoints for semantic segmentation using dilated convolutions.

## Contents

### Segmentation Datasets
Place your datasets here:

- `pascal_voc/` - PASCAL VOC 2012 segmentation dataset
- `cityscapes/` - Cityscapes urban scene dataset
- `coco/` - COCO dataset with segmentation annotations
- `ade20k/` - ADE20K scene parsing dataset
- `custom/` - Your custom segmentation datasets

### Audio Datasets (for WaveNet)
- `audio_samples/` - Audio files for WaveNet experiments
- `speech/` - Speech datasets

### Model Checkpoints
- `checkpoints/` - Saved model weights
- `pretrained/` - Pre-trained segmentation models
  - `deeplab_v3_*.pth`
  - `deeplab_v3plus_*.pth`
  - `wavenet_*.pth`

### Logs & Visualizations
- `tensorboard/` - TensorBoard logs
- `wandb/` - Weights & Biases logs
- `segmentation_results/` - Output segmentation masks
- `receptive_field_viz/` - Receptive field visualizations

## Downloading Datasets

### PASCAL VOC 2012
```python
from torchvision import datasets

# Download automatically
dataset = datasets.VOCSegmentation(
    root='./data/pascal_voc',
    year='2012',
    image_set='train',
    download=True
)
```

### COCO
```bash
# Download COCO 2017
cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

### Cityscapes (Requires Registration)
1. Register at https://www.cityscapes-dataset.com/
2. Download:
   - `leftImg8bit_trainvaltest.zip` (~11 GB)
   - `gtFine_trainvaltest.zip` (~241 MB)
3. Extract to `data/cityscapes/`

Structure:
```
cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

### Audio Data (for WaveNet)
```python
import librosa

# Load audio
audio, sr = librosa.load('data/audio_samples/sample.wav', sr=16000)
```

## Dataset Sizes

- **PASCAL VOC 2012**: ~2 GB
- **COCO 2017**: ~25 GB (images + annotations)
- **Cityscapes**: ~11 GB
- **ADE20K**: ~4 GB
- **Audio samples**: Varies (typically 10-100 MB per hour)

## Notes

- Segmentation datasets require significant storage
- Dilated convolutions maintain high resolution throughout
- Multi-scale training requires flexible input sizes
- For dense prediction, avoid excessive downsampling
