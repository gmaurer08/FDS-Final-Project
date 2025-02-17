# FDS - Final Project: Classification of Fake and Real Faces

This repository contains our group's final project in the *Fundamentals of Data Science and Laboratory* course for the master's degree in Data Science at Sapienza (2024/2025).

### Group Members
-  Nefeli Apostolou, 2168240
- Zaineb Ojayyan, 2182087
- Géraldine Valérie Maurer, 1996887
- Angelina Kholopova, 2168055
- Johanna Eklundh, 2183564

## Dataset
In this project, we use the 140k-real-and-fake-faces dataset sourced from https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces. The real images come from the [Flickr](https://github.com/NVlabs/ffhq-dataset) dataset collected by Nvidia , and the remaining half was sampled from the 1 Million fake faces generated by [StyleGAN](https://github.com/NVlabs/stylegan) and provided by Bojan on kaggle. The images contained in it are of size 256x256 pixels with color channels RGB, depicting faces at their center. When it comes to the second dataset we used for extra testing on the models we used the Labelled Faces in the Wild (LFW) Dataset (real images) from [https://www.kaggle.com/datasets/jessicali9530/lfw-dataset] and the Academic Dataset by Generated Photos.

## Project Steps
- **Baseline Models**: logistic regression and simple CNN
- **Custom Model**: CNN with custom architecture
- **Benchmark Model 1**: ResNet-50
- **Benchmark Model 2**: EfficientNet
- **Data Augmentation**: flipping, rotating, color changes, Gaussian blur
- **Grad-CAM**: explainability and finding key facial features

### Grad-CAM images links
* **CustomCNN Grad-CAM images**: https://drive.google.com/drive/folders/1m_nPV8dCLw2I0ioUPVHf7lzj9DgO1nQC
* **ResNet-50 Grad-CAM images**: https://drive.google.com/drive/folders/112aDaR8khSr4Rap0nvNGOEl_pn2nGRKG

## Licensing

### ResNet-50
This project uses the **ResNet-50 model** provided by **The MathWorks, Inc.** under the following license:

[BSD 3-Clause "New" or "Revised" License](LICENSE-BSD-3-Clause.txt)

### Copyright Notice
Copyright (c) 2019, The MathWorks, Inc.

### EfficientNet

This project uses the **EfficientNet model**, which is licensed under the [Apache License 2.0](LICENSE-Apache-License-Version-2.0.txt) by:

- **The TensorFlow Authors**
- **Pavel Yakubovskiy**

### Copyright Notice

Copyright © 2019 The TensorFlow Authors, Pavel Yakubovskiy.

### File Structure

```bash
├── CustomCNN/
│   ├── CustomCNN + ResNet50 results.csv
│   ├── CustomCNN.ipynb
│   ├── results_images/
│   │   ├── CustomCNN/
│   │   ├── CustomCNN2/
├── EfficientNet/
│   ├── checkpoint.pth
│   ├── checkpoints/
│   ├── checkpoints_latest/
│   ├── efficeintnet.py
│   ├── efficientnet_result/
│   ├── efficientnet_result_images/
│   ├── efficientnet_result_latest/
│   ├── gradcam_output/
│   ├── training_results.csv
├── Tests and Data Augmentation/
│   ├── Data_Augmentation_and_Grad_CAM.ipynb
│   ├── test_dataset.ipynb
│   ├── Testing_on_new_Images.ipynb
│   ├── Testing_with filters.ipynb
├── Grad-CAM/
│   ├── Grad-CAM-CustomCNN/
│   │   ├── true-fake-pred-fake/
│   │   ├── true-fake-pred-real/
│   │   ├── true-real-pred-fake/
│   │   ├── true-real-pred-real/
│   ├── Grad-CAM-resnet50/
│   │   ├── true-fake-pred-fake/
│   │   ├── true-fake-pred-real/
│   │   ├── true-real-pred-fake/
│   │   ├── true-real-pred-real/
├── LICENSE
├── LICENSE-Apache-License-Version-2.0.txt
├── LICENSE-BSD-3-Clause.txt
├── LogisticRegression_SimpleCNN.ipynb
├── README.md
├── getStructure.py
├── resnet50/
│   ├── resnet50.ipynb
│   ├── resnet50_results_images/
```
