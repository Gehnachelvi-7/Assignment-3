# ResNet Implementation and Comparison (Assignment 3)

This repository contains the implementation of the paper:

**Deep Residual Learning for Image Recognition (2015)**

The goal of this assignment is to implement a ResNet architecture from scratch and compare it with an official implementation.

---

## Repository Structure

### 🔹 custom_resnet.py
- Contains the **from-scratch implementation** of ResNet
- Includes:
  - Residual Block
  - Skip connections
  - Network architecture (SmallResNet)
- Core idea: replicate residual learning concept

---

### 🔹 train_custom.py
- Trains the custom ResNet model
- Includes:
  - Data loading (CIFAR-10)
  - Training loop
  - Loss computation
  - Accuracy evaluation
- Outputs:
  - `custom_model.pth`
  - `custom_results.txt`

---

### 🔹 official_eval.py
- Runs the **official ResNet-18 model (PyTorch)**
- Uses:
  - `torchvision.models.resnet18`
  - Pretrained weights (ImageNet)
- Fine-tunes on CIFAR-10
- Compares results with custom model

---

### 🔹 gnr_assignment_3.ipynb
- Google Colab notebook used for:
  - Running training
  - Using GPU
  - Saving results to Drive
- Contains complete execution pipeline

---

### 🔹 GNR_assignment_3_report.pdf
- Final report of the assignment
- Includes:
  - Paper explanation
  - Implementation details
  - Results
  - Comparison
  - Analysis

---

### 🔹 resnet_results/
Contains all output files:

- `custom_model.pth` → trained custom model  
- `custom_results.txt` → custom model accuracy, time, parameters  
- `extra_time.txt` → training time logs  
- `final_results.txt` → comparison of custom vs official model  

---

## Dataset Used

- **CIFAR-10**
  - 60,000 images
  - 10 classes
  - 32×32 resolution

### Why CIFAR-10?
The original paper uses ImageNet, but due to computational constraints, CIFAR-10 is used as a smaller alternative while keeping the same task (image classification).

---

## Methodology

### Custom Implementation
- Built from scratch
- Residual connections implemented manually
- Trained using Adam optimizer

### Official Implementation
- Used PyTorch ResNet-18
- Pretrained on ImageNet
- Fine-tuned on CIFAR-10

---

## Results

| Model | Accuracy | Parameters |
|------|---------|-----------|
| Custom ResNet | ~80% | ~2.7M |
| Official ResNet-18 | ~81% | ~11M |

---

## Key Observations

- Residual connections improve training stability  
- Pretrained models perform better due to transfer learning  
- Custom model achieves good performance with fewer parameters  
- Trade-off between efficiency and accuracy  

---

## Conclusion

This project demonstrates the effectiveness of residual learning.  
The custom implementation successfully captures the core idea of ResNet, while the official model shows the advantage of pretraining and deeper architectures.

---

## References

- Paper: https://arxiv.org/abs/1512.03385  
- Original Repo: https://github.com/KaimingHe/deep-residual-networks  
- PyTorch ResNet: https://github.com/pytorch/vision  
- Assignment Repo: https://github.com/Gehnachelvi-7/Assignment-3  

---
