import os
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from models.suim_net import SUIMNet  # Import your PyTorch version of the SUIMNet
from utils import trainDataGenerator  # Assume we have an equivalent function for DataLoader

# Dataset and checkpoint directories
dataset_name = "suim"
train_dir = "/home/liw324/code/data/SUIM_datasets/SUIM/train_val"
ckpt_dir = "SUIM/ckpt/"
base_ = 'VGG'  # or 'RSB'

if base_ == 'RSB':
    im_res_ = (320, 240, 3)
    ckpt_name = "suimnet_rsb.pth"
else:
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg.pth"

model_ckpt_name = os.path.join(ckpt_dir, ckpt_name)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
suimnet = SUIMNet(base=base_, im_res=(im_res_[1], im_res_[0]), n_classes=5)
suimnet = suimnet.to(device)

# Load saved model if needed
if os.path.exists(model_ckpt_name):
    suimnet.load_state_dict(torch.load(model_ckpt_name))

# Training parameters
batch_size = 8
num_epochs = 50
learning_rate = 1e-4

# Optimizer and scheduler
optimizer = optim.Adam(suimnet.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Loss function (binary cross-entropy as in the original code)
criterion = torch.nn.BCELoss()

# Data augmentation using torchvision.transforms
data_gen_args = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((im_res_[1], im_res_[0]), scale=(0.95, 1.05)),
    transforms.ToTensor()
])

# Dataset and DataLoader (implement trainDataGenerator as needed)
train_dataset = trainDataGenerator(train_dir, transform=data_gen_args)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Model checkpoint save function
def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

# Training loop
best_loss = float('inf')
for epoch in range(num_epochs):
    suimnet.train()
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = suimnet(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Scheduler step
    scheduler.step()

    # Print epoch loss
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Save model checkpoint if it improves
    if epoch_loss < best_loss:
        print(f'Saving model with loss {epoch_loss:.4f}')
        best_loss = epoch_loss
        save_checkpoint(suimnet, model_ckpt_name)