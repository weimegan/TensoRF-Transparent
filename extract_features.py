import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Transparent KeyPose Data
# descriptor_path = "data/descriptors-transparent" 
# image_path = "data/transparent/images_4"
# feature_path = "data/transparent-features/images_4"
# histogram_path = "data/histograms/transparent"

# img_file = f'{image_path}/000000_L.png'

# desc_file = f'{descriptor_path}/000000_L.pt'

# Flower Data
descriptor_path = "../../TensoRF/data/descriptors/images_4"
image_path = "../../TensoRF/data/nerf_llff_data/flower/images_4"
feature_path = "../../TensoRF/data/flower-features/images_4"
histogram_path = "../../TensoRF/data/histograms/flower"
# flower coordinate
# h, w = desc.shape[0]//2, desc.shape[1]//2
# leaf coordinate
# h, w = 5,5 

# Fixed descriptor
desc = torch.load(f'{descriptor_path}/image020.pt')
def select_features(file_name):
    """
    Input: filename without extension ex. 000000_L
            h, w coordinates of desired feature
    Output: image with selected features
    """
    img = Image.open(f'{image_path}/{file_name}.png')
    desc_img = torch.load(f'{descriptor_path}/{file_name}.pt')
    print("desc shape ", desc_img.shape) # 224, 398, 384

    h, w = desc.shape[0]//2, desc.shape[1]//2

    mid_desc = torch.mean(desc[h-2:h+2, w-2:w+2, :], dim=(0,1))#.reshape(1,1,desc.shape[-1])
    print("mid desc shape ", mid_desc.shape)
    print(mid_desc.mean(), mid_desc.min(), mid_desc.max(), mid_desc.sum())

    dot_out = torch.einsum("...k,ijk->ij", mid_desc, desc_img)
    print("dot out shape ", dot_out.shape)
    print(dot_out[h-2:h+2, w-2:w+2])
    print(dot_out.mean(), dot_out.min(), dot_out.max(), dot_out.sum())

    # Plot histogram to determine potential threshold
    plt.hist(dot_out)
    plt.savefig(f'{histogram_path}/hist-dot_out-{file_name}.png')

    new_size = (dot_out.shape[1], dot_out.shape[0])
    print(new_size)
    print("image size ", img.size)
    img_rescaled = np.asarray(img.resize(new_size))
    print(img_rescaled.shape)

    # Flower thresh = 10
    # Leaf thresh = 15-20
    thresh = 10
    dot_out = torch.unsqueeze(dot_out, -1)
    print(dot_out.shape, img_rescaled.shape)
    masked_dot_out = np.where((dot_out > thresh), img_rescaled, np.zeros((1,1,3))).astype(np.uint8)
    print(type(masked_dot_out))
    print(masked_dot_out.shape)
    print(masked_dot_out.min(), masked_dot_out.max())
    final_im = Image.fromarray(masked_dot_out)
    final_im.save(f'{feature_path}/{file_name}.png')

if __name__ == "__main__":
    for f in sorted(os.listdir(image_path)):
        file_name = f.split('.')[0]
        print(file_name)
        select_features(file_name)