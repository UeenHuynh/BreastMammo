# Mammogram Deep Learning Pipeline for Breast Cancer Detection

A comprehensive deep learning pipeline for mammogram classification supporting multiple datasets, model architectures, and advanced training techniques for breast cancer detection research.

## 🔬 Project Overview

This project implements a robust mammogram classification system designed for breast cancer detection using state-of-the-art deep learning techniques. The pipeline supports:

- **Binary Classification**: Benign vs. Malignant detection
- **Multi-class Classification**: Normal, Benign, Malignant classification
- **ROI Processing**: Region of Interest extraction and analysis
- **Transfer Learning**: Pre-trained models with two-phase training
- **Data Augmentation**: Advanced augmentation techniques including elastic transforms, mixup, and cutmix
- **Mixed Precision Training**: Optimized training with FP16 support

## 📊 Supported Datasets

| Dataset | Type | Classes | Image Size | Description |
|---------|------|---------|------------|-------------|
| **mini-MIAS** | Multi-class | Normal, Benign, Malignant | 1024×1024 | Mammographic Image Analysis Society database |
| **mini-MIAS-binary** | Binary | Benign, Malignant | 1024×1024 | Binary version of mini-MIAS |
| **CBIS-DDSM** | Binary | Benign, Malignant | 512×512 | Curated Breast Imaging Subset of DDSM |
| **CMMD** | Binary | Benign, Malignant | 224×224 | Chinese Mammography Database |
| **INbreast** | Multi-class | Normal, Benign, Malignant | 224×224 | INbreast database with BI-RADS classification |

### Dataset-Specific Features

- **CBIS-DDSM**: Supports mammogram type filtering (`calc`, `mass`, `all`)
- **INbreast**: BI-RADS mapping with automatic pectoral muscle removal
- **CMMD**: Optimized for binary classification with balanced augmentation

## 🏗️ Model Architectures

### Available Models

| Model | Type | Input Size | Description |
|-------|------|------------|-------------|
| **CNN** | Custom | Variable | Custom CNN architecture |
| **VGG** | Pre-trained | 224×224 or 1024×1024 | VGG19 with transfer learning |
| **VGG-common** | Pre-trained | 224×224 | Standard VGG19 configuration |
| **ResNet** | Pre-trained | 224×224 | ResNet50 architecture |
| **Inception** | Pre-trained | 224×224 | InceptionV3 model |
| **DenseNet** | Pre-trained | 224×224 | DenseNet121 architecture |
| **MobileNet** | Pre-trained | 224×224 | MobileNetV2 for efficient inference |

### Model-Specific Configurations

- **Custom CNN**: Adapts to input data dimensions automatically
- **Pre-trained Models**: Support two-phase training (frozen → unfrozen layers)
- **Input Channels**: 3 channels for pre-trained models, 1 channel for custom CNN

## 🛠️ Installation

### Requirements

```bash
pip install tensorflow>=2.8.0
pip install scikit-learn
pip install opencv-python
pip install pandas
pip install numpy
pip install matplotlib
pip install pydicom
pip install tensorflow-io
```

### Project Structure

```
mammogram-pipeline/
├── src/
│   ├── main.py                 # Main training script
│   ├── main2.py               # INbreast-optimized script
│   ├── mainruncmmd.py         # CMMD-optimized script
│   ├── config.py              # Configuration parameters
│   ├── cnn_models/            # Model architectures
│   ├── data_operations/       # Data preprocessing
│   └── utils.py               # Utility functions
├── data/                      # Dataset directory
├── saved_models/              # Trained model storage
└── output/                    # Results and visualizations
```

## 🚀 Usage Examples

### Basic Training Commands

#### CMMD Dataset with MobileNet
```bash
python src/main.py -d CMMD -m MobileNet -r train -b 8 -lr 1e-3 -e1 50 -e2 50
```

#### mini-MIAS with VGG
```bash
python src/main.py -d mini-MIAS -m VGG -r train -b 4 -lr 1e-4 -e1 100 -e2 50
```

#### CBIS-DDSM with ResNet (Mass only)
```bash
python src/main.py -d CBIS-DDSM -mt mass -m ResNet -r train -b 2 -lr 1e-3
```

#### INbreast with Pectoral Muscle Removal
```bash
python src/main2.py -d INbreast -m DenseNet -r train --remove_pectoral -b 4
```

### Testing Pre-trained Models

```bash
# Test CMMD model
python src/main.py -d CMMD -m MobileNet -r test

# Test with ROI processing
python src/main.py -d INbreast -m VGG -r test --roi
```

### Advanced Training Options

```bash
# With data augmentation and mixed precision
python src/main2.py -d INbreast -m MobileNet -r train \
  --apply_elastic --apply_mixup --remove_pectoral -b 8 -lr 1e-3

# Custom CNN with specific parameters
python src/main.py -d mini-MIAS -m CNN -r train -b 16 -lr 1e-2 -e1 200
```

## ⚙️ Configuration Parameters

### Key Parameters in `config.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate |
| `early_stopping_patience` | 10 | Epochs before early stopping |
| `reduce_lr_patience` | 5 | Epochs before LR reduction |
| `reduce_lr_factor` | 0.5 | LR reduction factor |
| `min_learning_rate` | 1e-6 | Minimum learning rate |
| `augment_data` | True | Enable data augmentation |

### Dataset-Specific Image Sizes

```python
MINI_MIAS_IMG_SIZE = {"HEIGHT": 1024, "WIDTH": 1024}
CMMD_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
INBREAST_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
VGG_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
MOBILE_NET_IMG_SIZE = {"HEIGHT": 224, "WIDTH": 224}
```

### BI-RADS Mapping (INbreast)

```python
BI_RADS_MAPPING = {
    "Normal": ["BI-RADS 1"],
    "Benign": ["BI-RADS 2", "BI-RADS 3"],
    "Malignant": ["BI-RADS 4a", "BI-RADS 4b", "BI-RADS 4c", "BI-RADS 5", "BI-RADS 6"]
}
```

## 📁 Main Scripts Overview

### `main.py` - General Purpose Script
- Supports all datasets and models
- Standard training pipeline
- Suitable for most experiments

### `main2.py` - INbreast Optimized
- Specialized for INbreast dataset
- Automatic pectoral muscle removal
- Mixed precision training
- Advanced augmentation options

### `mainruncmmd.py` - CMMD Optimized
- Optimized for CMMD dataset
- Efficient binary classification
- Streamlined preprocessing

## 🎯 Advanced Features

### ROI Processing
```bash
# Enable ROI extraction
python src/main.py -d INbreast -m VGG --roi
```

### Two-Phase Training
1. **Phase 1**: Frozen pre-trained layers (`max_epoch_frozen`)
2. **Phase 2**: Unfrozen fine-tuning (`max_epoch_unfrozen`)

### Data Augmentation
- Rotation, scaling, shearing
- Elastic deformation
- Mixup and CutMix techniques
- CLAHE enhancement

### Mixed Precision Training
Automatically enabled in `main2.py` for faster training and reduced memory usage.

## 🏋️ Training Process

### Standard Workflow
1. **Data Loading**: Dataset-specific preprocessing
2. **Model Creation**: Architecture selection and compilation
3. **Phase 1 Training**: Frozen pre-trained layers
4. **Phase 2 Training**: Fine-tuning all layers
5. **Evaluation**: Performance metrics and visualization

### Early Stopping & LR Scheduling
- Monitor validation loss for early stopping
- Reduce learning rate on plateau
- Save best model weights automatically

### Class Weight Handling
Automatic class weight calculation for imbalanced datasets to improve minority class performance.

## 📊 Model Evaluation

The pipeline provides comprehensive evaluation including:
- Accuracy and loss curves
- Confusion matrices
- Classification reports
- ROC curves and AUC scores
- Grad-CAM visualizations (where applicable)

## 🔬 Research Applications

This pipeline has been designed for:
- Comparative studies of deep learning architectures
- Transfer learning effectiveness analysis
- Data augmentation impact assessment
- ROI vs. full image classification comparison
- Multi-dataset generalization studies

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{mammogram_dl_pipeline,
  title = {Mammogram Deep Learning Pipeline for Breast Cancer Detection},
  abstract = {Breast cancer claims 11,400 lives on average every year in the UK, making it one of the deadliest diseases. This pipeline explores various deep learning techniques for mammogram classification using CNNs with transfer learning approaches.},
  license = {BSD-2-Clause},
  year = {2024}
}
```

## 📄 License

This project is licensed under the BSD-2-Clause License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the pipeline.

## 📞 Support

For questions or issues, please refer to the documentation or create an issue in the repository.