import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np  # Needed for proper dtype conversion

# Define CIFAR-10 classes
classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load CIFAR-10 dataset with grayscale conversion
transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Normalize grayscale images
x_train_gray = torch.stack([img for img, _ in trainset])

# Visualize grayscale images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train_gray[i][0], cmap='gray')  # Use grayscale channel
    plt.title(classes[trainset[i][1]])  # Get label
    plt.axis('off')

plt.show()

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Convert PyTorch tensor to NumPy uint8 format for SIFT
image_for_sift = (x_train_gray[0][0].numpy() * 255).astype(np.uint8)  # Proper conversion

# Detect keypoints and descriptors for one image
keypoints, descriptors = sift.detectAndCompute(image_for_sift, None)

# Draw keypoints
sift_img = cv2.drawKeypoints(image_for_sift, keypoints, None)

# Show image with SIFT keypoints
plt.figure(figsize=(5, 5))
plt.imshow(sift_img, cmap='gray')
plt.title("SIFT Keypoints")
plt.axis('off')
plt.show()
