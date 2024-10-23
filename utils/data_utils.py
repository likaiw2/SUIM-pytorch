import os
import fnmatch
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

"""
RGB color code and object categories:
------------------------------------
000 BW: Background waterbody
001 HD: Human divers
010 PF: Plants/sea-grass
011 WR: Wrecks/ruins
100 RO: Robots/instruments
101 RI: Reefs and invertebrates
110 FV: Fish and vertebrates
111 SR: Sand/sea-floor (& rocks)
"""
mask_type={"HD":1,      # HD: Human divers
           "PF":2,      # PF: Plants/sea-grass
           "WR":3,      # WR: Wrecks/ruins
           "RO":4,      # RO: Robots/instruments
           "RI":5,      # RI: Reefs and invertebrates
           "FV":6,      # FV: Fish and vertebrates
           "SR":7       # SR: Sand/sea-floor (& rocks)
           }      

def mask_code_to_image(mask_code):
    '''accept a 1-7 h*w array and turn it into h*w*rgb'''
    rows, cols = mask_code.shape
    bin_array = np.zeros((rows, cols, 3), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            binary_str = np.binary_repr(mask_code[i, j], width=3)
            # print(mask_code[i, j],":",binary_str)
            bin_array[i, j] = [int(bit) for bit in binary_str]
            
    return 255*bin_array

def image_to_mask(rgb_mask):
    '''convert h*w*rgb into 1-7 array'''
    rgb_mask=rgb_mask/255.0
    new_list=np.zeros([rgb_mask.shape[0],rgb_mask.shape[1]])
    for i in range(rgb_mask.shape[0]):
        for j in range(rgb_mask.shape[1]):  
            binary_string = ''.join(str(int(x)) for x in rgb_mask[i][j])
            new_list[i][j] = int(binary_string, 2)
    return new_list

class SUIMDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, sal=False, target_size=(256, 256,3)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        
        self.img_list = sorted(os.listdir(img_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        assert len(self.img_list)==len(self.mask_list),"the number of img and mask not paired"
        
        self.transform = transform
        self.sal = sal
        self.target_size = target_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # Open image and masks
        img = Image.open(f"{self.img_dir}/{self.img_list[idx]}")
        mask = Image.open(f"{self.mask_dir}/{self.mask_list[idx]}")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        img = np.array(img)/255.0
        mask = np.array(mask)
   
        mask=image_to_mask(mask)
        human_mask = np.where(mask == mask_type["HD"], 1, 0)
        plant_mask = np.where(mask == mask_type["PF"], 1, 0)
        wreck_mask = np.where(mask == mask_type['WR'], 1, 0)
        robot_mask = np.where(mask == mask_type['RO'], 1, 0)
        reef_mask = np.where(mask == mask_type['RI'], 1, 0)
        fish_mask = np.where(mask == mask_type['FV'], 1, 0)
        sand_mask = np.where(mask == mask_type['SR'], 1, 0)
        
        masks=np.array([human_mask,
                        plant_mask,
                        wreck_mask,
                        robot_mask,
                        reef_mask,
                        fish_mask,
                        sand_mask
                        ])
        
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).float()
        masks_tensor = torch.from_numpy(masks).float()

        return img_tensor, masks_tensor


if __name__=="__main__":
    train_dir = "/home/liw324/code/data/SUIM_datasets/SUIM/train_val"
    img_dir = f"{train_dir}/images"
    mask_dir = f"{train_dir}/masks"
    
    # data_gen_args = transforms.Compose([
    #     transforms.RandomRotation(20),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomResizedCrop((255,255), scale=(0.95, 1.05)),
    # ])
    
    # train_dataset=SUIMDataset(img_dir=img_dir,mask_dir=mask_dir,transform=data_gen_args)
    # # print(len(train_dataset))
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    # for i,item in enumerate(train_loader):
    #     print(item)