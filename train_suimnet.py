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
from datetime import datetime
from tqdm import tqdm


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


# parameters
dataset_name = "SUIM"
# dataset_name = "SUIM_lowlight"
# dataset_name = "SUIM_lowlight_enhance"

train_dir = f"/home/liw324/code/data/SUIM_datasets/{dataset_name}/train_val"
ckpt_dir = f"/home/liw324/code/SUIM-pytorch/ckpt/{current_time}_{dataset_name}"

img_dir = f"{train_dir}/images"
mask_dir = f"{train_dir}/masks"

# model_ckpt_name = "/home/liw324/code/SUIM-pytorch/ckpt/#best/low_light_fine_tune_epoch_16_loss0.1581_acc_97.09524972098214.pth"
model_ckpt_name = ""

device = torch.device("cuda:0")
batch_size = 16
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
    wandb.init(project="suim-pytorch",
               config={"base_":base_,
                       "batch_size":batch_size,
                       "num_epochs":num_epochs,
                       "learning_rate":learning_rate,
                       "train_size":train_size,
                       "val_size" :val_size,
                       "device":device,
                       "start_time":current_time,
                       "dataset":dataset_name}
               )
    
wandb_init()
    
# Running loop
best_loss = float('inf')
best_accuracy = 0
for epoch in tqdm(range(num_epochs),unit="epoch"):
    # Training
    suimnet.train() 
    running_loss = 0.0
    for i, (images, masks,_) in enumerate(train_loader):
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
    sum_acc=0
    for i, (images, masks, _) in enumerate(val_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = suimnet(images)
        
        # print("outputs: ",outputs.shape)
        # print("masks: ",masks.shape)
        
        loss = criterion(outputs, masks)

        val_loss += loss.item()
        
        # Find the most powerful feature
        max_values, _ = torch.max(outputs, dim=1)
        max_values = torch.unsqueeze(max_values, dim=1)
        outputs = torch.where(outputs == max_values, outputs, torch.zeros_like(outputs))
        
        # Make present to 0 or 1
        bin_outputs = (outputs >= 0.5).int()
        
        # Calculate acc
        accuracy = (bin_outputs == masks).sum().item() / outputs.numel()
        sum_acc += accuracy
        
        wandb.log({'val_iter_loss': loss.item(),
                   'val_acc': accuracy})
        
    # Scheduler step
    scheduler.step()

    # Print epoch loss
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * sum_acc / len(val_loader)
    
    print(f'train_loss: {epoch_loss:.4f}        val_loss: {val_loss / len(val_loader):.4f}        epoch_accuracy: {epoch_acc:.4f}')
    wandb.log({'train_epoch_loss':epoch_loss,
               'val_epoch_loss':val_loss / len(val_loader),
               'epoch_accuracy':epoch_acc
               })

    # Save model checkpoint if it improves
    if epoch_loss < best_loss:
        print(f'Saving model with loss {epoch_loss:.4f}')
        best_loss = epoch_loss
        save_checkpoint(suimnet, f"{ckpt_dir}/{dataset_name}_epoch_{epoch}_loss{epoch_loss:.4f}_acc_{epoch_acc:.4f}.pth")
    if epoch_acc > best_accuracy:
        print(f'Saving model with acc {accuracy}')
        best_accuracy = epoch_acc
        save_checkpoint(suimnet, f"{ckpt_dir}/{dataset_name}_epoch_{epoch}_loss{epoch_loss:.4f}_acc_{epoch_acc:.4f}.pth")