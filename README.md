# Convolutional Neural Network Project

## 🚀 Overview

This repository contains a Convolutional Neural Network (CNN) project focused on various aspects of model training, hyper-parameter tuning, and advanced learning techniques. The project systematically investigates the effects of hyper-parameter adjustments, dropout testing, data augmentation, and few-shot learning using prototypical networks. We suggest you check the full [report](https://github.com/kichal01/DL_Project1/blob/main/The_report.pdf) for a detailed explanation.

## 📌 About the project

The project uses a **CINIC10** dataset consisting of 270,000 images. It was split into equal subsets: train, validation, and test, each containing 90,000 images. The tests were conducted partly on NVIDIA GeForce GTX 1650 and partly on Google Colab’s GPU T4. The CNNs were trained solely using the **PyTorch** library.

## Influence of Hyper-parameter Change Related to Training Process

### Learning rate

The experiments evaluated learning rates for EfficientNet-B0 and ResNet-50, finding that optimal rates vary by model. EfficientNet achieved ~64% accuracy, with signs of overfitting. ResNet adapted with layer modifications, showed faster convergence but unclear superiority. Dropout improved generalization, while early stopping limited training duration. Results suggest model-specific tuning and potential benefits from longer training and advanced augmentation.

### Dropout (with Soft-voting ensemble)

Three custom CNNs were tested with different dropout rates to study overfitting and underfitting. Training used Adam (lr=0.003), batch size=64, and 50 epochs. Model 1 performed best, balancing training and validation loss, while Model 2 showed slight overfitting and Model 3 underfitting. A soft-voting ensemble was then tested with different weightings, improving test accuracy by ~2% over individual models (75.94% accuracy). The results confirm that higher dropout values in later layers help generalization, and weighted ensembles can enhance performance.

## Influence of Data Augmentations

### Standard Data Augmentations

The study evaluated various augmentation techniques (rotations, grayscale, scaling, and combinations) on EfficientNet (both pretrained and non-pretrained). Small rotations (±10°) performed best, while extreme rotations (±180°) hurt performance; background color (black vs. default) had minimal impact. Pretrained models worked best with minimal augmentation (scaling matched default performance), suggesting pretraining reduces augmentation benefits. Non-pretrained models struggled (max ~60% accuracy), likely due to insufficient training. Mixed augmentation (random rotations/scaling at 20–75% probability) improved results but still underperformed vs. default transforms. Grayscale consistently degraded performance. Key takeaway: Moderate, selective augmentation (e.g., light rotations/scaling) helps, but excessive or grayscale transformations harm model accuracy. Default data often outperforms augmented sets, especially for pretrained models.

### More Advanced Data Augmentations

Our tests with CutMix augmentation showed its effectiveness depends heavily on model complexity. When applied to EfficientNet, it achieved 70% validation accuracy in just 12 epochs - outperforming other methods despite higher loss values from its multi-class approach. Simple scaling also worked well, while other techniques matched baseline performance.
However, results differed with a simpler custom CNN. Here, CutMix actually reduced performance (61% vs 63% with other methods), suggesting it's better suited for complex architectures. The technique clearly benefits advanced models but proves unnecessary for simpler networks, where basic augmentation works best. Scaling remained consistently effective across both architectures.

## Prototypical Networks - Few-shot Learning Technique

The few-shot learning experiments revealed some compelling patterns. Using a pretrained DenseNet121 with prototypical networks, we achieved 64.5% accuracy on the full 90k-image dataset - impressive for 30-shot, 10-way classification. The model showed particular strength in distinguishing dogs from similar classes, thanks to its clever embedding-space separation. However, we observed sudden validation loss spikes from vanishing gradients.
When we reduced the training data to 30k images, performance dipped to 58% accuracy. The smaller dataset led to clear overfitting after 2000 episodes and amplified gradient issues.
These results highlight two important tradeoffs: while prototypical networks excel at separating visually similar classes through embedding optimization, they require substantial data to avoid overfitting and gradient problems. The approach works best when you have enough quality examples to learn those crucial class relationships in the embedding space.

## ⚙️ Installation and Usage

In order to reproduce results for files Project1_fewshot_new.ipynb and Dropout_Softvoting.ipynb:
1. Download the dataset as .zip file from 'https://www.kaggle.com/datasets/mengcius/cinic10'.
2. Upload it to Google Colab.
3. Make a new code-block with !unzip archive.zip command.
   
In order to reproduce results for files DL_Project1.ipynb, DataAugmentation.ipynb, LearningRate.ipynb, plots.ipynb and Simple_CNN.ipynb:
1. Put the files and dataset into one folder
(current path to dataset used in every one of these files is 'cinic10/versions/1/').

## 👨‍💻 Contributors

- [Michał Korwek](https://github.com/kichal01)
- [Emil Łasocha](https://github.com/emilook86)
