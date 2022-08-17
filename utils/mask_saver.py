"""
project: deep learning for infected cell segmentation

author Information
==================
name: amzad hossain rafi
email: amzad.rafi@northsouth.edu
github:

"""




from rle_to_mask import rle2mask 
from PIL import Image ,ImageOps
import os   
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt



train_csv_dir = "B:\keggle segmentation\hubmap-organ-segmentation\\train.csv"
train_dir = "B:\keggle segmentation\hubmap-organ-segmentation\\train_images\\"
mask_dir = "B:\keggle segmentation\hubmap-organ-segmentation\\train_masks\\"


def create_mask_folder():
    """
    arg : 
    retrun : 
    """
    
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    else: 
        pass
    return None
    
    
def read_csv(path):
    """
    Reads the csv file and returns the data in a pandas dataframe.
    :param path: The path to the csv file.
    :return: The data in the csv file.
    """
    return pd.read_csv(path)    




def save_mask(mask_dir,  mask):
    """
    this funtion take rle value and convert to mask and save the mask in mask_dir
    arg : mask_dir, mask_name, mask
    return : None
    """

    list_shape=[]
    img_name=[]
    
    mata_data=read_csv(path=train_csv_dir)
    
    for i in os.listdir(train_dir):
        name=img_name.append(i.split('.')[0])
        read_img =  Image.open(train_dir+i)
        shape1=np.array(read_img)
        list_shape.append(shape1.shape[:2])


    for i in range(len(mata_data['rle'])):
        rle_to_array=rle2mask(mata_data['rle'][i],shape=list_shape[i])
        plt.imsave(mask_dir+img_name[i]+".png",rle_to_array)

        #convet to pil image instance
        # array_to_img=Image.fromarray(rle_to_array.astype('uint8'), 'RGB')
        
        # array_to_img.save(mask_dir+mask+'_'+str(i)+'.png')


if __name__ == '__main__':
    create_mask_folder()
    save_mask(mask_dir=mask_dir, mask='mmask')

    



    

