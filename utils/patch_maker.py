"""
project: deep learning for infected cell segmentation

author Information
==================
name: amzad hossain rafi
email: amzad.rafi@northsouth.edu
github:

"""




import os
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt


patch_size = 256

train_dir = "B:\keggle segmentation\hubmap-organ-segmentation\\train_images\\"
mask_dir = "B:\keggle segmentation\hubmap-organ-segmentation\\train_masks\\"
patch_mask_dir = "B:\keggle segmentation\hubmap-organ-segmentation\\train_masks_patch\\"
patch_img_dir = "B:\keggle segmentation\hubmap-organ-segmentation\\train_images_patch\\"



def creat_img_mask_patch_folder():
    """
    create train_images_patch and train_masks_patch folder

    arg : none
    retrun : none 
    """
    
    if not os.path.exists(patch_mask_dir):
        os.makedirs(patch_mask_dir)
    if not os.path.exists(patch_img_dir):
        os.makedirs(patch_img_dir)
            
    else: 
        pass
    return None


def patch_maker_img():
    img_naming_counter=0
    

    for i in os.listdir(train_dir):
        read_img = Image.open(train_dir+i)
        shape1=np.array(read_img)
        crop_size = shape1.shape[0]//patch_size*patch_size
        resized_img = read_img.crop((0,0,crop_size,crop_size))
        img_to_array = np.array(resized_img)
        patch_len= (shape1.shape[0]//patch_size)-1


        for i in range(patch_len): 
            for j in range(patch_len):
                x=img_to_array[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                #save the patch image
                plt.imsave(patch_img_dir+str(img_naming_counter)+".png",x)
                img_naming_counter+=1
    return None


def patch_maker_mask():
    mask_naming_counter = 0 


    
    for i in os.listdir(mask_dir):
        read_img = Image.open(mask_dir+i)
        shape1=np.array(read_img)
        crop_size = shape1.shape[0]//patch_size*patch_size
        crop_img = read_img.crop((0,0,crop_size,crop_size))
        img_to_array = np.array(crop_img)
        patch_len= (shape1.shape[0]//patch_size)-1


        for i in range(patch_len): 
            for j in range(patch_len):
                x=img_to_array[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size]
                #save the patch image
                imge=Image.fromarray(x)
                imge.save(patch_mask_dir+str(mask_naming_counter)+".png")
                #plt.imsave(patch_mask_dir+str(mask_naming_counter)+".png",x)
                mask_naming_counter+=1
    return None

# if __name__ == '__main__':
#     patch_maker_img()
#     creat_img_mask_patch_folder()
#     patch_maker_mask()
#     print("done")



       

