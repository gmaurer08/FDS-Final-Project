# FDS - Final Project: Classification of Fake and Real Faces

This repository contains our group's final project in the *Fundamentals of Data Science and Laboratory* course for the master's degree in Data Science at Sapienza (2024/2025).

## Group Members
-  Nefeli Apostolou, 2168240
- Zaineb Ojayyan, 2182087
- Géraldine Valérie Maurer, 1996887
- Angelina Kholopova, 2168055
- Johanna Eklundh, 2183564

### Project Plan

- **Baseline**: Implement logistic regression and/or simple neural network (MLP or CNN) baseline model  
- **Advanced**: Implement CNN with custom architecture (difficulty: task requires a complex CNN, which is hard to build from scratch)  
- **Pretrained Model 1**: Fine-tune ResNet-50 to the dataset  
- **Pretrained Model 2**: Fine-tune EfficientNet to the dataset
- **Data Augmentation + Filters**: Experiment with augmented images (crop, flip, rotate, add noise, etc.) and Gaussian filtered images
- **Generalize**: Find another dataset to test the models on new data (because this dataset only has StyleGAN generated data), fine-tune if necessary

### Objectives

- Implement baseline and advanced models for binary image classification  
- Use pre-trained models for efficiency, given time and resource constraints  
- Achieve high accuracy with pre-trained fine-tuned models  
- Test robustness to data augmentation (if time permits)
- Provide explainable results with Grad-CAM
- Compare accuracy and computational complexity of all models
