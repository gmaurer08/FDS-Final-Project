# This script uses the EfficientNet model provided by The TensorFlow Authors and Pavel Yakubovskiy.
# Licensed under the Apache License 2.0. See LICENSE-Apache-License-Version-2.0.txt for details.
#
# Copyright (c) 2019 The TensorFlow Authors, Pavel Yakubovskiy.

import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
from PIL import Image
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import seaborn as sns
import cv2
from tabulate import tabulate
import zipfile
import copy
print(torch.cuda.is_available())  # This will return True if a GPU is available.
import pickle

torch.manual_seed(42)
np.random.seed(42)
random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# # !!!!Dont run this if you already have the dataset downloaded!!!!
# !kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# # !!!! Dont run this if you already have the dataset downloaded and unzipped!!!!

# # Specify the path to your zip file and the destination folder
# zip_file_path = "../../dataset/140k-real-and-fake-faces.zip"  # Ensure this matches your downloaded file's name
# destination_folder = "../../dataset/extracted_files"  # The folder where the content will be extracted

# # Unzip the file
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(destination_folder)

# print(f"Dataset extracted to {destination_folder}")


dataset_path = "../dataset/extracted_files/"  # Updated path, one folder out, then dataset
train_df = pd.read_csv(dataset_path +"train.csv", index_col=0)
test_df = pd.read_csv(dataset_path +"test.csv", index_col=0)
valid_df = pd.read_csv(dataset_path +"valid.csv", index_col=0)
# print('Train Dataset Paths and Labels')
# display(train_df.head())
# print('Test Dataset Paths and Labels')
# display(test_df.head())
# print('Validation Dataset Paths and Labels')
# display(valid_df.head())

# Base directory where images are stored
base_img_dir = '../dataset/extracted_files/real_vs_fake/real-vs-fake/'

# Combine the base path with the relative paths from the 'path' column
train_df['image_path'] = train_df['path'].apply(lambda x: os.path.join(base_img_dir, x))
valid_df['image_path'] = valid_df['path'].apply(lambda x: os.path.join(base_img_dir, x))
test_df['image_path'] = test_df['path'].apply(lambda x: os.path.join(base_img_dir, x))

# Pick 3 real and 3 fake images to show
real_imgs_to_show = random.sample(list(train_df[train_df.label == 1].image_path), 3)
fake_imgs_to_show = random.sample(list(train_df[train_df.label == 0].image_path), 3)

#C:\Users\Asus\Desktop\Utlandsstudier\Kurser\FDS\FinalProject\dataset\extracted_files\real_vs_fake\real-vs-fake\train\real 
# Open images using PIL
real_images = [Image.open(path) for path in real_imgs_to_show]
fake_images = [Image.open(path) for path in fake_imgs_to_show]

# Display images side by side using Matplotlib
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Add real images in the first row
for ax, img, title in zip(axes[0], real_images, ["Real Image 1", "Real Image 2", "Real Image 3"]):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

# Add fake images in the second row
for ax, img, title in zip(axes[1], fake_images, ["Fake Image 1", "Fake Image 2", "Fake Image 3"]):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

class FaceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        '''
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels
            transform (callable, optional): Optional transform to be applied on a sample
        '''
        self.dataframe = dataframe  # Store the dataframe
        self.transform = transform  # Store the transform

    def __len__(self):
        return len(self.dataframe)  # Return the length of the dataframe

    def __getitem__(self, idx):
    
        # Get the image path and label from the dataframe
        img_path = self.dataframe.iloc[idx].image_path
        label = int(self.dataframe.iloc[idx].label)  # Label is 0 or 1 (fake or real)

        # Open the image using PIL
        image = Image.open(img_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label, img_path # Return image and label

# Load pre-trained EfficientNet-B0 (It is possible to change to B1, B2, etc., if desired) 
# B0 is the smallest and fastest, while B7 is the largest and most accurate.
model = models.efficientnet_b0(pretrained=True)

# Define the image transformations for EfficientNet
efficientnet_transform = transforms.Compose([
    transforms.Resize(256),  # Resize images to 256px
    transforms.CenterCrop(224),  # Crop the center 224x224 part
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Create Dataset instances for train, validation, and test sets with EfficientNet transformation
train_dataset = FaceDataset(dataframe=train_df, transform=efficientnet_transform)
valid_dataset = FaceDataset(dataframe=valid_df, transform=efficientnet_transform)
test_dataset = FaceDataset(dataframe=test_df, transform=efficientnet_transform)

# Create DataLoader instances for efficient batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check the first batch of data
data_iter = iter(train_loader)
images, labels, img_path = next(data_iter)
print('Batch Shape:')
print(images.shape)  # Should print (batch_size, 3, 224, 224)
print('\nNumber of Labels:')
print(labels.shape)  # Should print (batch_size=64)

# Define the number of classes in your dataset
NUM_CLASSES = 2  # Real vs Fake
IMG_SIZE = 224

# Load the pre-trained EfficientNetB0
def build_model(num_classes):
    # Use the weights parameter instead of pretrained=True
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1  # Specify the desired weights
    model = efficientnet_b0(weights=weights)

    # Freeze the pre-trained layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier with a custom one
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),  # Regularization
        nn.Linear(model.classifier[1].in_features, num_classes)  # Adjust output for num_classes
    )

    return model

# Initialize the model
model = build_model(NUM_CLASSES)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
def train_model(model, train_loader, valid_loader, test_loader, loss_function, optimizer, lr, weight_decay = 0, momentum = 0, num_epochs=1, start_epoch=0, checkpoint_path='checkpoint.pth', result_path="efficientnet_result/results.pkl"):
    '''
    Function that performs training, validation and testing on a model, given a loss function, optimizer, and hyper-parameters.
    The function returns the results of training the model
    Args:
    - model (nn.Module): Model to train
    - train_loader (DataLoader): DataLoader for training set
    - valid_loader (DataLoader): DataLoader for validation set
    - test_loader (DataLoader): DataLoader for test set
    - num_epochs (int): Number of epochs to train for
    - start_epoch (int): Which epoch to start on
    - checkpoint_path (string): Where to store the checkpoints for continue training
    - loss_function (nn.Module): Loss function to use
    - optimizer (torch.optim): Optimizer to use
    - lr (float): Learning rate for optimizer
    - weight_decay (float): Weight decay for optimizer
    - momentum (float): Momentum for optimizer
    
    Returns:
    - model (nn.Module): Trained model
    - results (dict): Dictionary containing training and validation loss and accuracy for each epoch + other results
    '''
    # Initialize lists to store metrics
    train_losses, train_accuracies, val_losses, val_accuracies, true_labels, pred_labels, probs = [], [], [], [], [], [], []

    # Define Loss function
    if loss_function == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
    elif loss_function == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss

    # Define optimizer function
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Adam optimizer
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum) # SGD optimizer


    num_epochs += start_epoch  # Train for additional epoch

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        total_train = 0
        prob = []
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Set parameter gradients to zero
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)

            # Compute loss
            if loss_function == 'BCEWithLogitsLoss':
                # convert labels to float and ensure single-channel output
                loss = criterion(outputs[:, 1], labels.float())
            elif loss_function == 'CrossEntropyLoss':
                loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            # Optimization
            optimizer.step()

            # Calculate metrics
            train_running_loss += loss.item()
            preds = torch.max(outputs, 1)[1]
            train_correct += torch.sum(preds == labels.data)
            total_train += labels.size(0)


        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_acc = train_correct / total_train
        train_losses.append(epoch_train_loss) # Save training loss for this epoch
        train_accuracies.append(epoch_train_acc.item())  # Save training accuracy for this epoch

        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}")

        # Validation phase
        model.eval()
        valid_running_loss = 0.0
        valid_correct = 0
        valid_total = 0

        # Initialize label lists for this epoch
        epoch_true_labels = []
        epoch_pred_labels = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)

                # Compute loss
                if loss_function == 'BCEWithLogitsLoss':
                    val_loss = criterion(outputs[:, 1], labels.float()) # convert labels to float and ensure single-channel output 
                elif loss_function == 'CrossEntropyLoss':
                    val_loss = criterion(outputs, labels)
                
                valid_running_loss += val_loss.item()
                
                val_preds = torch.max(outputs, 1)[1]
                valid_correct += (val_preds == labels).sum().item()
                valid_total += labels.size(0)

                # Memorize true and predicted labels for explainability
                epoch_true_labels.extend(labels.cpu().numpy())
                epoch_pred_labels.extend(val_preds.cpu().numpy())

                 # Calculate probabilities for prediction = class 1 (real) using softmax
                prob.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy())

        epoch_valid_loss = valid_running_loss / len(valid_loader)
        epoch_valid_acc = valid_correct / valid_total
        val_losses.append(epoch_valid_loss)  # Save validation loss for this epoch
        val_accuracies.append(epoch_valid_acc)  # Save validation accuracy for this epoch

        # Save results to the probs, true_labels, pred_labels lists
        probs.append(prob)
        true_labels.append(epoch_true_labels)
        pred_labels.append(epoch_pred_labels)

        print(f"Valid Loss: {epoch_valid_loss:.4f}, Valid Accuracy: {epoch_valid_acc:.4f}")

        # Save the checkpoint at the end of each epoch to be able to continue the training later
        torch.save({
            'epoch': epoch +1, # Save the next epoch for proper resumption
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': epoch_valid_loss  # Save validation loss for reference
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")
    
    print('Finished Training')

    # confusion matrix
    conf_matrix = confusion_matrix(true_labels[-1], pred_labels[-1])

    # Get predictions on the test set
    model.eval()

    with torch.no_grad(): # no need to compute gradients here

        test_predictions = []
        test_true_labels = []

        # Iterate over the batches in test_loader
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) # move to device
            outputs = model(images) # compute output
            _, predicted = torch.max(outputs, 1) # get prediction
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(test_true_labels, test_predictions)
    precision = precision_score(test_true_labels, test_predictions, average='binary')
    recall = recall_score(test_true_labels, test_predictions, average='binary')
    f1 = f1_score(test_true_labels, test_predictions, average='binary')
    roc_auc = roc_auc_score(test_true_labels, test_predictions)

    # Print the results
    print('\nTest Results')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print(f'Test ROC AUC: {roc_auc:.4f}')

    # Test Confusion Matrix
    test_conf_matrix = confusion_matrix(test_true_labels, test_predictions)

    report = classification_report(true_labels[-1], pred_labels[-1], target_names=['Fake', 'Real'])
    print("\nClassification Report:\n", report)

    # After training, save the metrics into a DataFrame for visualization or further analysis
    efficientnet_results = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'probs': probs,
        'test_true_labels': test_true_labels,
        'test_predictions': test_predictions,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'conf_matrix': conf_matrix,
        'test_conf_matrix': test_conf_matrix
    }

    # Save the efficientnet_results dictionary to a file
    with open(result_path, 'wb') as f:
        pickle.dump(efficientnet_results, f)

    print("Training results saved to ", result_path)

    return model, efficientnet_results

def test_model(model, test_loader):
    '''
    Function to test a model on a test dataset.
    '''
    print('start ')

    #print("Model Parameters Checksum:", sum(p.sum().item() for p in model.parameters()))

    model.eval()
    test_predictions, test_true_labels = [], []
    accuracy, precision, recall, f1, roc_auc = 0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(test_true_labels, test_predictions)
    precision = precision_score(test_true_labels, test_predictions, average='binary')
    recall = recall_score(test_true_labels, test_predictions, average='binary')
    f1 = f1_score(test_true_labels, test_predictions, average='binary')
    roc_auc = roc_auc_score(test_true_labels, test_predictions)

    print('\nTest Results')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print(f'Test ROC AUC: {roc_auc:.4f}')

    test_conf_matrix = confusion_matrix(test_true_labels, test_predictions)
    print("\nConfusion Matrix:\n", test_conf_matrix)

    return {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_roc_auc': roc_auc,
        'confusion_matrix': test_conf_matrix,
    }

def print_result (results):

    train_accuracies = results['train_accuracies']
    val_accuracies = results['val_accuracies']
    recall = results['recall']
    f1 = results['f1']
    roc = results['roc_auc']

    train_accuracies_last = train_accuracies[4]
    val_accuracies_last= val_accuracies[4]
    print('Train Accuracy: ', train_accuracies_last, '\nValid Accuracy: ', val_accuracies_last, '\nRecall: ', recall, '\nF1: ', f1, '\nRoc: ', roc)

def get_grad_cam_images(model, transform, real_images, fake_images, path):
    '''
    Function that takes a trained model and returns Grad-CAM results for a selection of real and fake image paths
    Args:
    - model (nn.Module): Trained model
    - transform (torchvision.transforms): Image transformations
    - target_layer (nn.Module): Target layer for Grad-CAM
    - real_images (list): List of real image paths
    - fake_images (list): List of fake image paths
    - path (str): Path to save Grad-CAM images
    Returns:
    - None
    '''
    def get_last_conv_layer(model):
        """Retrieve the last convolutional layer for Grad-CAM."""
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d)
                return module
        raise ValueError("No Conv2d layer found in the model.")
   
    target_layer = model.features[0][0] 

    #Set the model to evaluation mode
    model.eval()
    for param in model.parameters():
        param.requires_grad = true

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    target_layer.register_forward_hook(lambda module, input, output: output.retain_grad())

    # Function to process an image and generate Grad-CAM heatmap
    def generate_gradcam_overlay(img_path, target_class):
        
        image = Image.open(img_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Convert the image to a NumPy array for visualization
        img_for_visualization = np.array(image.resize((224, 224))) / 255.0  # Scale to [0, 1]
        with torch.enable_grad():
            # Generate Grad-CAM heatmap
            output = model(img_tensor)
            target = torch.zeros_like(output)
            target[0, target_class] = 1  # One-hot encode the target class
            output.backward(gradient=target)

            target_layer.register_forward_hook(
                lambda module, input, output: output.requires_grad_()
            )

            try:
                grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(target_class)])
                print("Grad-CAM computed successfully.")

            except Exception as e:
                raise RuntimeError(f"Error during Grad-CAM computation: {e}")
    
        if grayscale_cam is None:
            raise RuntimeError("Grad-CAM returned None.")
        heatmap = grayscale_cam[0, :]  # Extract the heatmap

        # Overlay heatmap on the image
        superimposed_img = show_cam_on_image(img_for_visualization, heatmap, use_rgb=True)

        # Get model prediction
        with torch.no_grad():
            # output = model(img_tensor)
            pred_class = output.argmax(dim=1).item()

        return superimposed_img, pred_class

    # Create subplots
    fig, axes = plt.subplots(2, len(real_images), figsize=(15, 10))

    # First row: Real images
    for i, img_path in enumerate(real_images):
        target_class = 1  # Actual label for real images
        superimposed_img, pred_class = generate_gradcam_overlay(img_path, target_class)
        axes[0, i].imshow(superimposed_img)
        axes[0, i].set_title(f"Real Image {i+1}\nPred: {pred_class}, Actual: {target_class}")
        axes[0, i].axis("off")

    # Second row: Fake images
    for i, img_path in enumerate(fake_images):
        target_class = 0  # Actual label for fake images
        superimposed_img, pred_class = generate_gradcam_overlay(img_path, target_class)
        axes[1, i].imshow(superimposed_img)
        axes[1, i].set_title(f"Fake Image {i+1}\nPred: {pred_class}, Actual: {target_class}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def plot_figures(results, path, num_epochs):
    '''
    Function that, given training, validation and test results, plots and saves images
    Args:
    - results (dict): Dictionary containing training, validation and test results
    - path (str): Path to save images
    - num_epochs (int): Number of epochs
    Returns:
    - None
    '''
    train_losses = results['train_losses']
    val_losses = results['val_losses']
    train_accuracies = results['train_accuracies']
    val_accuracies = results['val_accuracies']
    true_labels = results['true_labels']
    pred_labels = results['pred_labels']
    probs = results['probs']
    conf_matrix = results['conf_matrix']
    test_conf_matrix = results['test_conf_matrix']

    plt.figure(figsize=(15, 16))

    # Training and validation loss curves
    plt.subplot(3, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Trainining Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    #plt.savefig(path[0])

    # Training and validation accuracy curves
    plt.subplot(3, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    #plt.savefig(path[1])

    # ROC curve
    fpr, tpr, _ = roc_curve(true_labels[-1], probs[-1])

    plt.subplot(3, 2, 3)
    plt.plot(fpr, tpr, color='blue', label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid()
    #plt.savefig(path[2])

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels[-1], probs[-1])

    plt.subplot(3, 2, 4)
    plt.plot(recall, precision, color='green', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    #plt.savefig(path[3])

    # Confusion Matrix Validation

    # Plot the confusion matrix
    plt.subplot(3,2,5)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True, xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
    plt.title('Validation Confusion Matrix (last epoch)')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    #plt.savefig(path[4])

    # Confusion Matrix Testing
    plt.subplot(3,2,6)
    sns.heatmap(test_conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True, xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(path[0])
    #plt.show()
    plt.close()
    
## To conitune training ##
# #Load previous epochs (training)
# checkpoint = torch.load('checkpoint.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# #start_epoch = checkpoint['epoch'] + 1  # This should be 1 for the second epoch
# print("Starting training from epoch:", start_epoch)

## Restarting the training ##
# Start fresh: Reset model weights and optimizer state
# model.apply(lambda module: module.reset_parameters() if hasattr(module, 'reset_parameters') else None)  # Reset model weights
# #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reinitialize optimizer
# start_epoch = 0 # Ensure we start from epoch 0
# print("Starting training from epoch:", start_epoch)

os.makedirs('Efficientnet/checkpoints', exist_ok=True)
os.makedirs('Efficientnet/efficientnet_result', exist_ok=True)

def run_experiment ():
    #Create a copy of efficientNet
    efficientnet_BCE_Adam_001 = copy.deepcopy(model)
    efficientnet_BCE_Adam_001, results_BCE_Adam_001 = train_model(efficientnet_BCE_Adam_001, train_loader, valid_loader, test_loader, loss_function = 'BCEWithLogitsLoss', optimizer = 'Adam', lr = 0.001, weight_decay = 0, momentum = 0, num_epochs=5, start_epoch=0, checkpoint_path='checkpoints/checkpoint_BCE_Adam_001.pth', result_path='efficientnet_result/result_BCE_Adam_001.pth')

    # Create a copy of efficientNet
    efficientnet_CE_Adam_01 = copy.deepcopy(model)
    efficientnet_CE_Adam_01, results_CE_Adam_01 = train_model(efficientnet_CE_Adam_01, train_loader, valid_loader, test_loader, loss_function = 'CrossEntropyLoss', optimizer = 'Adam', lr = 0.01, weight_decay = 0, momentum = 0, num_epochs=5, start_epoch=0, checkpoint_path='checkpoints/checkpoint_CE_Adam_01.pth', result_path='efficientnet_result/result_CE_Adam_01.pth')

    # Create copy of efficientnet
    efficientnet_BCE_SGD_001 = copy.deepcopy(model)
    efficientne_BCE_SGD_001, results_BCE_SGD_001 = train_model(efficientnet_BCE_SGD_001, train_loader, valid_loader, test_loader, loss_function = 'BCEWithLogitsLoss', optimizer = 'SGD', lr = 0.001, weight_decay = 0, momentum = 0, num_epochs=5, start_epoch=0, checkpoint_path='checkpoints/checkpoint_BCE_SGD_001.pth', result_path='efficientnet_result/result_BCE_SGD_001.pth')

    # Create copy of efficientnet   DOESNT EXIST IN RESNET50
    efficientnet_BCE_Adam_01 = copy.deepcopy(model)
    efficientne_BCE_Adam_01, results_BCE_Adam_01 = train_model(efficientnet_BCE_Adam_01, train_loader, valid_loader, test_loader, loss_function = 'BCEWithLogitsLoss', optimizer = 'Adam', lr = 0.001, weight_decay = 0, momentum = 0, num_epochs=5, start_epoch=0, checkpoint_path='checkpoints/checkpoint_BCE_Adam_01.pth', result_path='efficientnet_result/result_BCE_Adam_01.pth')

    # Create copy of efficientnet
    efficientnet_CE_SDG_001_mom = copy.deepcopy(model)
    efficientnet_CE_SDG_001_mom, results_CE_SDG_001_mom = train_model(efficientnet_CE_SDG_001_mom, train_loader, valid_loader, test_loader, loss_function = 'CrossEntropyLoss', optimizer = 'SGD', lr = 0.001, weight_decay = 0, momentum = 0.9, num_epochs=5, start_epoch=0, checkpoint_path='checkpoints/checkpoint_CE_SDG_001_mom.pth', result_path='efficientnet_result/result_CE_SDG_001_mom.pth')

    # Create copy of efficientnet
    efficientnet_BCE_SGD_0001_mom = copy.deepcopy(model)
    efficientnet_BCE_SGD_0001_mom, results_BCE_SGD_0001_mom = train_model(efficientnet_BCE_SGD_0001_mom, train_loader, valid_loader, test_loader, loss_function = 'BCEWithLogitsLoss', optimizer = 'SGD', lr = 0.0001, weight_decay = 0, momentum = 0.99, num_epochs=5, start_epoch=0, checkpoint_path='checkpoints/checkpoint_BCE_Adam_001_wd.pth', result_path='efficientnet_result/result_BCE_Adam_001_wd.pth')

def load_experiment(result_path):
    with open(result_path, 'rb') as f:
        loaded_results = pickle.load(f)
    print("succsessfully loaded: " + result_path)
    #print("Loaded training results:", loaded_results)
    return loaded_results

def results ():
    ### Load results from experiments ###
    results_BCE_Adam_001 = load_experiment('Efficientnet/efficientnet_result_latest/result_BCE_Adam_001.pkl')
    results_CE_Adam_01 = load_experiment('Efficientnet/efficientnet_result_latest/result_CE_Adam_01.pkl')
    results_BCE_SGD_001 = load_experiment('Efficientnet/efficientnet_result_latest/result_BCE_SGD_001.pkl')
    results_BCE_Adam_01 = load_experiment('Efficientnet/efficientnet_result_latest/result_BCE_Adam_01.pkl')
    results_CE_SGD_001_mom = load_experiment('Efficientnet/efficientnet_result_latest/result_CE_SGD_001_mom.pkl')
    results_BCE_SGD_0001_mom = load_experiment('Efficientnet/efficientnet_result_latest/result_BCE_SGD_0001_mom.pkl')

    ### CREATE IMAGES WITH THE RESULTS ###
    #BCE_Adam_001
    BCE_Adam_001_paths_plots = ['Efficientnet/efficientnet_result_Images/BCE_Adam_001.png'];
    plot_figures(results_BCE_Adam_001, BCE_Adam_001_paths_plots, 5)
    # BCE_SGD_0001_mom
    BCE_SGD_0001_mom_paths_plots = ['Efficientnet/efficientnet_result_Images/BCE_SGD_0001_mom.png'];
    plot_figures(results_BCE_SGD_0001_mom, BCE_SGD_0001_mom_paths_plots, 5)
    # BCE_Adam_01
    BCE_Adam_01_paths_plots = ['Efficientnet/efficientnet_result_Images/BCE_Adam_01.png'];
    plot_figures(results_BCE_Adam_01, BCE_Adam_01_paths_plots, 5)
    # BCE_SGD_001
    BCE_SGD_001_paths_plots = ['Efficientnet/efficientnet_result_Images/BCE_SGD_001.png'];
    plot_figures(results_BCE_SGD_001, BCE_SGD_001_paths_plots, 5)
    # CE_Adam_01 
    CE_Adam_01_paths_plots = ['Efficientnet/efficientnet_result_Images/CE_Adam_01.png'];
    plot_figures(results_CE_Adam_01, CE_Adam_01_paths_plots, 5)
    # CE_SGD_001_mom
    CE_SGD_001_mom_paths_plots = ['Efficientnet/efficientnet_result_Images/CE_SGD_001_mom.png'];
    plot_figures(results_CE_SGD_001_mom, CE_SGD_001_mom_paths_plots, 5)

    ### Print results ###
    #BCE_Adam_001_paths = ['Efficientnet/efficientnet_result_Images/Table_BCE_Adam_001.png'];
    print ('BCE_Adam_001: ')
    print_result(results_BCE_Adam_001)

    #CE_Adam_01_paths = ['Efficientnet/efficientnet_result_Images/Table_CE_Adam_01.png'];
    print ('CE_Adam_01: ')
    print_result(results_CE_Adam_01)

    #BCE_SGD_001_paths = ['Efficientnet/efficientnet_result_Images/Table_SGD_Adam_001.png'];
    print ('BCE_Adam_001: ')
    print_result(results_BCE_SGD_001)

    #BCE_Adam_01_paths = ['Efficientnet/efficientnet_result_Images/Table_BCE_Adam_01.png'];
    print ('BCE_Adam_001: ')
    print_result(results_BCE_Adam_01)

    #CE_SGD_001_mom_paths = ['Efficientnet/efficientnet_result_Images/Table_CE_SGD_001_mom.png'];
    print ('CE_SGD_001_mom: ')
    print_result(results_CE_SGD_001_mom)

    #BCE_SGD_0001_mom_paths = ['Efficientnet/efficientnet_result_Images/Table_BCE_SGD_0001_mom.png'];
    print ('BCE_SGD_0001_mom: ')
    print_result(results_BCE_SGD_0001_mom)

def gradcam_and_testing():
    print('length ', len(test_loader))

    # BCE_Adam_001
    checkpoint_path = "EfficientNet/checkpoints_latest/checkpoint_BCE_Adam_001.pth"  # Path to your saved model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # Load model weights
    model.to(device)
    efficientnet_BCE_Adam_001 = copy.deepcopy(model)

    # CE_Adam_01 
    checkpoint_path = "EfficientNet/checkpoints_latest/checkpoint_CE_Adam_01.pth"  # Path to your saved model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # Load model weights
    model.to(device)
    efficientnet_CE_Adam_01  = copy.deepcopy(model)

    # BCE_SGD_001
    checkpoint_path = "EfficientNet/checkpoints_latest/checkpoint_BCE_SGD_001.pth"  # Path to your saved model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # Load model weights
    model.to(device)
    efficientnet_BCE_SGD_001 = copy.deepcopy(model)

    # BCE_Adam_01
    checkpoint_path = "EfficientNet/checkpoints_latest/checkpoint_BCE_Adam_01.pth"  # Path to your saved model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # Load model weights
    model.to(device)
    efficientnet_BCE_Adam_01 = copy.deepcopy(model)

    # CE_SGD_001_mom
    checkpoint_path = "EfficientNet/checkpoints_latest/checkpoint_CE_SGD_001_mom.pth"  # Path to your saved model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # Load model weights
    model.to(device)
    efficientnet_CE_SGD_001_mom = copy.deepcopy(model)
    
    # BCE_SGD_0001_mom 
    checkpoint_path = "EfficientNet/checkpoints_latest/checkpoint_BCE_SGD_0001_mom.pth"  # Path to your saved model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # Load model weights
    model.to(device)
    efficientnet_BCE_SGD_0001_mom = copy.deepcopy(model)

    ## GradCam ###
    get_grad_cam_images(efficientnet_BCE_Adam_001, efficientnet_transform, real_images, fake_images, output_path) # Generate Grad-CAM results
    output_path = "EfficientNet/gradcam_output/gradcam_results_BCE_Adam_001.png" # Path to save Grad-CAM output
    get_grad_cam_images(efficientnet_BCE_SGD_0001_mom, efficientnet_transform, real_images, fake_images, output_path) # Generate Grad-CAM results
    output_path = "EfficientNet/gradcam_output/gradcam_results_BCE_SGD_0001_mom.png" # Path to save Grad-CAM output
    output_path = "EfficientNet/gradcam_output/gradcam_results_BCE_Adam_01NEW_TEST.png" # Path to save Grad-CAM output
    get_grad_cam_images(efficientnet_BCE_Adam_01, efficientnet_transform, real_images, fake_images, output_path) # Generate Grad-CAM results
    get_grad_cam_images(efficientnet_BCE_SGD_001, efficientnet_transform, real_images, fake_images, output_path) # Generate Grad-CAM results
    output_path = "EfficientNet/gradcam_output/gradcam_results_BCE_SGD_001.png" # Path to save Grad-CAM output
    get_grad_cam_images(efficientnet_CE_Adam_01 , efficientnet_transform, real_images, fake_images, output_path) # Generate Grad-CAM results
    output_path = "EfficientNet/gradcam_output/gradcam_results_CE_Adam_01.png" # Path to save Grad-CAM output
    get_grad_cam_images(efficientnet_CE_SGD_001_mom, efficientnet_transform, real_images, fake_images, output_path) # Generate Grad-CAM results
    output_path = "EfficientNet/gradcam_output/gradcam_results_CE_SDG_001_mom.png" # Path to save Grad-CAM output
    
    ## Testing ###
    test_result_BCE_Adam_001 = test_model(efficientnet_BCE_Adam_001, test_loader)
    print('BCE_Adam_001: ')
    test_result_CE_Adam_01 = test_model(efficientnet_CE_Adam_01, test_loader)
    print('CE_Adam_01')
    test_result_BCE_SGD_001 = test_model(efficientnet_BCE_SGD_001, test_loader)
    print('BCE_SGD_001')
    test_result_BCE_Adam_01 = test_model(efficientnet_BCE_Adam_01, test_loader)
    print('BCE_Adam_01')
    test_result_CE_SGD_001_mom = test_model(efficientnet_CE_SGD_001_mom, test_loader)
    print('CE_SGD_001_mom')
    test_result_BCE_SGD_0001_mom = test_model(efficientnet_BCE_SGD_0001_mom, test_loader)
    print('BCE_SGD_0001_mom')

    ## Save GRADCAM with all different options ###
    # Filter images
    # Assuming dataloader is created from a DataFrame-based dataset
    case_00_paths, case_11_paths = filter_image_paths_by_case(efficientnet_CE_SGD_001_mom, test_loader, device)

    # Define output paths
    output_path_correct = "EfficientNet/gradcam_output/gradcam_CE_SGD_001_mom_features[0][0].png"

    ### Grad-CAM Visualization ###
    # Correct cases (real_images and fake_images)
    get_grad_cam_images(efficientnet_CE_SGD_001_mom, efficientnet_transform, case_11_paths, case_00_paths, output_path_correct)

def filter_image_paths_by_case(model, dataloader, device):
    """
    Filters image paths into four lists based on prediction and true label combinations.
    Returns exactly 2 image paths per case: (0,0), (1,1).
    """
    model.eval()
    
    # Initialize lists for each case
    case_00_paths = []  # Pred = 0, Label = 0
    case_11_paths = []  # Pred = 1, Label = 1

    with torch.no_grad():
        for batch in dataloader:
            # Assuming dataloader returns (images, labels, paths)
            images, labels, paths = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(images)):
                pred_label = preds[i].item()
                true_label = labels[i].item()
                img_path = paths[i]  # Get the image path

                # Correctly predicted fake (0,0)
                if pred_label == 0 and true_label == 0 and len(case_00_paths) < 4:
                    case_00_paths.append(img_path)
                    print(f"Saved path to case_00 (Pred=0, True=0): {img_path}")

                # Correctly predicted real (1,1)
                elif pred_label == 1 and true_label == 1 and len(case_11_paths) < 4:
                    case_11_paths.append(img_path)
                    print(f"Saved path to case_11 (Pred=1, True=1): {img_path}")

            # Stop early if all cases have 2 image paths
            if len(case_00_paths) >= 4 and len(case_11_paths) >= 4:
                break

    print('Finished filtering image paths.')
    return case_00_paths, case_11_paths

# run_experiment()
# results()
gradcam_and_testing()
# print_results_terminal()
