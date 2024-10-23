import os
import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from models.suim_net import SUIMNet  # Import your PyTorch version of the SUIMNet
from utils.data_utils import SUIMDataset  # Assume we have an equivalent function for DataLoader
from tqdm import tqdm

# parameters
dataset_name = "suim"
train_dir = "/home/liw324/code/data/SUIM_datasets/SUIM/train_val"
img_dir = f"{train_dir}/images"
mask_dir = f"{train_dir}/masks"
ckpt_dir = "/home/liw324/code/SUIM-pytorch/ckpt/"
device = torch.device("cuda:1")
batch_size = 32
num_epochs = 50
learning_rate = 1e-4
base_ = 'VGG'
# base_ = 'RSB'

if base_ == 'RSB':
    im_res_ = (320, 240, 3)
    ckpt_name = "suimnet_rsb"
else:
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg"

model_ckpt_name = os.path.join(ckpt_dir, ckpt_name)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

suimnet = SUIMNet(base=base_, im_res=(im_res_[1], im_res_[0]), n_classes=7)
suimnet = suimnet.to(device)

# Load saved model if needed
if os.path.exists(model_ckpt_name):
    suimnet.load_state_dict(torch.load(model_ckpt_name))

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
])

# Dataset and DataLoader 
dataset = SUIMDataset(img_dir,mask_dir, transform=data_gen_args)
train_size = int(0.8 * len(dataset))  # 80% for train
val_size = len(dataset) - train_size  # 20% for test
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Model checkpoint save function
def save_checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def wandb_init():
    wandb.init(project="suim-pytorch")
    wandb.base_=base_
    wandb.batch_size = 32
    wandb.num_epochs = 50
    wandb.learning_rate = 1e-4
    wandb.train_size = train_size
    wandb.val_size = val_size
    wandb.device=device
    
wandb_init()
    
# Running loop
best_loss = float('inf')
best_accuracy = 0
for epoch in tqdm(range(num_epochs),unit="epoch"):
    # Training
    suimnet.train() 
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = suimnet(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        wandb.log({'train_iter_loss': loss.item()})
        
    # Validating
    suimnet.eval()  
    val_loss = 0.0
    correct = 0
    total = 0
    for i, (images, masks) in enumerate(val_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = suimnet(images)
        loss = criterion(outputs, masks)

        val_loss += loss.item()
        wandb.log({'val_iter_loss': loss.item()})

        _, predicted = torch.max(outputs.data, 1)
        total += masks.size(0)
        correct += (predicted == masks).sum().item()
        
    # Scheduler step
    scheduler.step()

    # Print epoch loss
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'train_loss: {epoch_loss:.4f}        val_loss: {val_loss / len(val_loader):.4f}        accuracy: {accuracy}')
    wandb.log({'train_epoch_loss':epoch_loss,
               'val_epoch_loss':val_loss / len(val_loader),
               'accuracy':accuracy
               })

    # Save model checkpoint if it improves
    if epoch_loss < best_loss:
        print(f'Saving model with loss {epoch_loss:.4f}')
        best_loss = epoch_loss
        save_checkpoint(suimnet, f"{model_ckpt_name}_epoch_{epoch}_loss{epoch_loss:.4f}_acc_{accuracy}.pth")
    if accuracy > best_accuracy:
        print(f'Saving model with acc {accuracy}')
        best_accuracy = accuracy
        save_checkpoint(suimnet, f"{model_ckpt_name}_epoch_{epoch}_loss{epoch_loss:.4f}_acc_{accuracy}.pth")