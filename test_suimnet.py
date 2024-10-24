import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from models.suim_net import SUIMNet  # Import your PyTorch version of the SUIMNet
from utils.data_utils import SUIMDataset, mask_code_to_image  # Assume we have an equivalent function for DataLoader
from datetime import datetime
from tqdm import tqdm
from PIL import Image

mask_type={"HD":1,      # HD: Human divers
           "PF":2,      # PF: Plants/sea-grass
           "WR":3,      # WR: Wrecks/ruins
           "RO":4,      # RO: Robots/instruments
           "RI":5,      # RI: Reefs and invertebrates
           "FV":6,      # FV: Fish and vertebrates
           "SR":7       # SR: Sand/sea-floor (& rocks)
           }  


# parameters
dataset_name = "SUIM"
# dataset_name = "SUIM_lowlight"
# dataset_name = "SUIM_lowlight_enhance"

model_ckpt_name = "/home/liw324/code/SUIM-pytorch/ckpt/#best/suim_epoch_19_loss0.1633_acc_0.0.pth"

test_dir = f"/home/liw324/code/data/SUIM_datasets/{dataset_name}/TEST"
img_dir = f"{test_dir}/images"
mask_dir = f"{test_dir}/masks"

out_dir = f"/home/liw324/code/SUIM-pytorch/test_output/{dataset_name}"
device = torch.device("cuda:0")
batch_size = 16
base_ = 'VGG'
im_res_ = (320, 256, 3)


# Create output directories if they don't exist
os.makedirs(out_dir, exist_ok=True)
for cate in mask_type:
    os.makedirs(f"{out_dir}/{cate}", exist_ok=True)

# Model configuration
base_ = 'VGG'  # or 'RSB'
im_res_ = (320, 256, 3)


# 加载模型
suimnet = SUIMNet(base=base_, im_res=(im_res_[1], im_res_[0]), n_classes=7)
suimnet = suimnet.to(device)
suimnet.load_state_dict(torch.load(model_ckpt_name))  # load the pretrained model
suimnet.eval()

# Dataset
data_gen_args = transforms.Compose([
    transforms.Resize((im_res_[1], im_res_[0])),  # only resize for test
])

# Dataset and DataLoader 
dataset = SUIMDataset(img_dir, mask_dir, transform=data_gen_args)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def evaluate_model(model, test_loader, device):
    model.eval()
    sum_acc=0
    with torch.no_grad():
        for images, masks, img_names in tqdm(test_loader, unit="pic"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            max_values, _ = torch.max(outputs, dim=1)  # 形状为 [height, width]
            max_values = torch.unsqueeze(max_values, dim=1)
            
            outputs = torch.where(outputs == max_values, outputs, torch.zeros_like(outputs))
            bin_outputs = (outputs >= 0.5).int()
            
            accuracy = (bin_outputs == masks).sum().item() / outputs.numel()
            sum_acc += accuracy
            
            bin_outputs = bin_outputs.cpu().numpy()
            
            for item_in_each_batch,img_name in zip(bin_outputs,img_names):
                img_name = img_name + '.bmp'
                merge_mask=np.zeros_like(item_in_each_batch[0])
                # print(mask_type)
                for category,bin_mask in zip(mask_type,item_in_each_batch):
                    mask = mask_type[category]*bin_mask
                    mask_image = mask_code_to_image(mask).astype(np.uint8)

                    img_path = f"{out_dir}/{category}/{img_name}"
                    Image.fromarray(mask_image).save(img_path)
                    
                    merge_mask = merge_mask+mask
                    
                merge_mask = mask_code_to_image(merge_mask).astype(np.uint8)
                img_path = f"{out_dir}/{img_name}"
                Image.fromarray(merge_mask).save(img_path)
                
                
            
    avg_acc = 100 * sum_acc/ len(test_loader)
    print(avg_acc)

# run the test
evaluate_model(suimnet, test_loader, device)