# %% [markdown]
# # Fine tuning EfficientNet

# %% [markdown]
# #### Import libraries

# %%
#%pip install torchcam

# %%
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#from torchvision.transforms import InterpolationMode
from torchvision import transforms, models
from PIL import Image
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics import roc_curve, auc
import seaborn as sns
print(torch.cuda.is_available())  # This will return True if a GPU is available.

# %% [markdown]
# Set random seeds for reproducibility of results.

# %%
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% [markdown]
# #### Import and read dataset

# %%
# !!!!Dont run this if you already have the dataset downloaded!!!!
# !kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# %%
# !!!! Dont run this if you already have the dataset downloaded and unzipped!!!!
# import zipfile

# # Specify the path to your zip file and the destination folder
# zip_file_path = "../dataset/140k-real-and-fake-faces.zip"  # Ensure this matches your downloaded file's name
# destination_folder = "../dataset/extracted_files"  # The folder where the content will be extracted

# # Unzip the file
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(destination_folder)

# print(f"Dataset extracted to {destination_folder}")


# %%
dataset_path = "../dataset/extracted_files/"  # Updated path, one folder out, then dataset

# %%
train_df = pd.read_csv("../dataset/extracted_files/train.csv", index_col=0)
test_df = pd.read_csv("../dataset/extracted_files/test.csv", index_col=0)
valid_df = pd.read_csv("../dataset/extracted_files/valid.csv", index_col=0)
print('Train Dataset Paths and Labels')
display(train_df.head())
print('Test Dataset Paths and Labels')
display(test_df.head())
print('Validation Dataset Paths and Labels')
display(valid_df.head())

# %% [markdown]
# Adjust paths in the `path` columns

# %%
# Base directory where images are stored
base_img_dir = '../dataset/extracted_files/real_vs_fake/real-vs-fake/'

# Combine the base path with the relative paths from the 'path' column
train_df['image_path'] = train_df['path'].apply(lambda x: os.path.join(base_img_dir, x))
valid_df['image_path'] = valid_df['path'].apply(lambda x: os.path.join(base_img_dir, x))
test_df['image_path'] = test_df['path'].apply(lambda x: os.path.join(base_img_dir, x))

# %% [markdown]
# #### Visualize some images

# %%
# Pick 3 real and 3 fake images to show
real_imgs_to_show = random.sample(list(train_df[train_df.label == 1].image_path), 3)
fake_imgs_to_show = random.sample(list(train_df[train_df.label == 0].image_path), 3)

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

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Create FaceDataset class
# We create a FaceDataset class that can be passed to a DataLoader, which will take care of batching and shuffling.

# %%
class FaceDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        '''
        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths and labels
            transform (callable, optional): Optional transform to be applied on a sample
        '''
        self.dataframe = dataframe  # Store the dataframe
        self.transform = transform  # Store the transform (if present)

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

        return image, label  # Return image and label

# %% [markdown]
# ## Fine-Tuning EfficientNet to the Dataset

# %% [markdown]
# #### Load EfficientNet
# 
# We will use the pre-trained CNN EfficientNet-B0 and fine-tune it on the 140k-real-and-fake-faces dataset, then compare its accuracy to that of other models. The EfficientNet-B0 model is pre-trained on ImageNet, and we will use its weights available through PyTorch's torchvision library. EfficientNet expects images that are pre-processed with specific transformations, which are detailed in the official PyTorch documentation for EfficientNet models: EfficientNet in PyTorch.
# 
# The pre-processing includes resizing the images, cropping them to a standard size, converting them to tensors, and normalizing them using the mean and standard deviation values appropriate for EfficientNet. We apply these transformations to ensure that the input data is compatible with the pre-trained EfficientNet model, allowing it to achieve optimal performance when fine-tuned on our specific dataset.

# %%
import torchvision.models as models

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

# %%
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
images, labels = next(data_iter)
print('Batch Shape:')
print(images.shape)  # Should print (batch_size, 3, 224, 224)
print('\nNumber of Labels:')
print(labels.shape)  # Should print (batch_size,)

# %% [markdown]
# #### Fine-Tune the Model

# %% [markdown]
# Load the pre-trained model

# %%
# Define the number of classes in your dataset
NUM_CLASSES = 2  # Real vs Fake
IMG_SIZE = 224

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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

# %%
# Initialize the model
model = build_model(NUM_CLASSES)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

# %% [markdown]
# #### Train the model

# %%
# Training loop
def train_model(model, train_loader, valid_loader, num_epochs=1, start_epoch=0, checkpoint_path='checkpoint.pth'): #Change num_epochs to 10 again
    num_epochs += start_epoch  # Train for one additional epoch
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct.double() / total_train

        print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.4f}")

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        total_valid = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Calculate metrics
                valid_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                valid_correct += torch.sum(preds == labels.data)
                total_valid += labels.size(0)

        epoch_valid_loss = valid_loss / total_valid
        epoch_valid_acc = valid_correct.double() / total_valid

        print(f"Valid Loss: {epoch_valid_loss:.4f}, Valid Accuracy: {epoch_valid_acc:.4f}")

        # Save the checkpoint at the end of each epoch
        torch.save({
            'epoch': epoch +1, # Save the next epoch for proper resumption
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': epoch_valid_loss  # Save validation loss for reference
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")


#Load previous epochs (training)
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#start_epoch = checkpoint['epoch'] + 1  # This should be 1 for the second epoch
start_epoch = 1
print("Starting training from epoch:", start_epoch)

# Train the model
train_model(model, train_loader, valid_loader, num_epochs=1, start_epoch=start_epoch, checkpoint_path='checkpoint.pth') #Change num_epochs to 10 again

# %% [markdown]
# #### Fine-Tune All Layers: 
# Once the classifier is trained, unfreeze all layers and train the entire model with a lower learning rate.

# %%
# Unfreeze all layers
for param in model.features.parameters():
    param.requires_grad = True

# Re-define optimizer with a lower learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Continue training the model
train_model(model, train_loader, valid_loader, num_epochs=2)

# %%


# %%


# %%
# Generate confusion matrix
# conf_matrix = confusion_matrix(true_labels, pred_labels)

# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True,
#             xticklabels=['Fake','Real'], yticklabels=['Fake','Real'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()

# %%
# report = classification_report(true_labels, pred_labels, target_names=['Class 0', 'Class 1'])
# print("Classification Report:\n", report)

# %%
# num_images = 5  # Number of images to display
# fig, axes = plt.subplots(1, num_images, figsize=(15, 5))  # Create a subplot with 1 row and 'num_images' columns

# for i in range(num_images):
#     # Display the image
#     axes[i].imshow(cam_images[2*i])  # Display each Grad-CAM image
#     axes[i].axis('off')  # Turn off axis
#     axes[i].set_title(f"Image {i+1}")  # Set a title for each image

# plt.tight_layout()  # Ensure that images are neatly arranged
# plt.show()


