# Nail diseases Classifier

This project implements a hybrid deep learning model for classifying nail diseases images. It combines a ResNet50 feature extractor with a Vision Transformer (ViT) classifier to achieve high accuracy in nail condition classification.

## Features

- Hybrid model architecture: ResNet50 + Vision Transformer
- Data augmentation techniques for improved generalization
- Learning rate scheduling and early stopping for optimal training
- Performance visualization (loss and accuracy plots)
- Confusion matrix generation for model evaluation

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- timm
- scikit-learn
- matplotlib
- numpy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/nail-image-classifier.git
   cd nail-image-classifier
   ```

2. Install the required packages:
   ```
   pip install torch torchvision timm scikit-learn matplotlib numpy
   ```

3. Prepare your dataset:
   - Organize your nail images into subdirectories within a 'Nail Dataset' folder
   - Each subdirectory should represent a class (e.g., healthy, fungal, etc.)

## Usage

1. Update the `dataset_dir` variable in the script to point to your dataset location.

2. Run the training script:
   ```
   python nail_classifier.py
   ```

3. The script will:
   - Train the model
   - Save the trained model as 'hybrid_image_classifier.pth'
   - Generate and save performance plots (loss_vs_epochs.png, accuracy_vs_epochs.png)
   - Create and save a confusion matrix (confusion_matrix.png)

## Customization

- Adjust hyperparameters such as learning rate, batch size, and number of epochs in the script to fine-tune performance.
- Modify the data augmentation pipeline in the `transform` variable to suit your specific dataset needs.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

```
Copyright 2024 Your Name

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Acknowledgments

- This project uses the timm library for the Vision Transformer implementation.
- The ResNet50 model is pretrained on ImageNet.

