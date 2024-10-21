import os
import ntpath
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from suim_net import SUIMNet  # Import your PyTorch version of SUIM-Net
from utils.data_utils import get_paths  # Assume this function is already converted

# Experiment directories
test_dir = "/home/liw324/code/data/SUIM_datasets/SUIM/TEST"
samples_dir = "data/test/output/"
RO_dir = samples_dir + "RO/"
FB_dir = samples_dir + "FV/"
WR_dir = samples_dir + "WR/"
HD_dir = samples_dir + "HD/"
RI_dir = samples_dir + "RI/"

# Create output directories if they don't exist
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(RO_dir, exist_ok=True)
os.makedirs(FB_dir, exist_ok=True)
os.makedirs(WR_dir, exist_ok=True)
os.makedirs(HD_dir, exist_ok=True)
os.makedirs(RI_dir, exist_ok=True)

# Model configuration
base_ = 'VGG'  # or 'RSB'
if base_ == 'RSB':
    im_res_ = (320, 240, 3)
    ckpt_name = "suimnet_rsb5.pth"
else:
    im_res_ = (320, 256, 3)
    ckpt_name = "suimnet_vgg5.pth"

# Initialize model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
suimnet = SUIMNet(base=base_, im_res=(im_res_[1], im_res_[0]), n_classes=5)
suimnet = suimnet.to(device)
suimnet.load_state_dict(torch.load(join("ckpt/saved/", ckpt_name)))
suimnet.eval()

# Input/output shapes
im_h, im_w = im_res_[1], im_res_[0]

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((im_h, im_w)),
    transforms.ToTensor()
])

def test_generator():
    # Test all images in the directory
    assert os.path.exists(test_dir), "Local image path doesn't exist"
    
    for p in get_paths(test_dir):
        # Read and scale inputs
        img = Image.open(p)
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            out_img = suimnet(img_tensor)
        
        # Thresholding
        out_img = (out_img > 0.5).float()
        
        print(f"Tested: {p}")
        
        # Get filename
        img_name = ntpath.basename(p).split('.')[0] + '.bmp'
        
        # Save individual output masks
        out_img = out_img.squeeze(0).cpu().numpy()
        ROs = np.uint8(out_img[0, :, :] * 255)
        FVs = np.uint8(out_img[1, :, :] * 255)
        HDs = np.uint8(out_img[2, :, :] * 255)
        RIs = np.uint8(out_img[3, :, :] * 255)
        WRs = np.uint8(out_img[4, :, :] * 255)
        
        Image.fromarray(ROs).save(RO_dir + img_name)
        Image.fromarray(FVs).save(FB_dir + img_name)
        Image.fromarray(HDs).save(HD_dir + img_name)
        Image.fromarray(RIs).save(RI_dir + img_name)
        Image.fromarray(WRs).save(WR_dir + img_name)

# Test images
test_generator()