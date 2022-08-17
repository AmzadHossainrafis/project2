import os 


patch_mask_dir = "B:\keggle segmentation\hubmap-organ-segmentation\\train_masks_patch\\"
patch_img_dir = "B:\keggle segmentation\hubmap-organ-segmentation\\train_images_patch\\"



def make_img_mask_list():
    """
    make a list of image and mask path
    arg : none
    retrun : none 
    """
    img_list = []
    mask_list = []
    for i in os.listdir(patch_img_dir):
        img_list.append(os.path.join(patch_img_dir+i))
    for i in os.listdir(patch_mask_dir):
        mask_list.append(os.path.join(patch_mask_dir+i))
    return img_list, mask_list


if __name__ == '__main__':
    x,y=make_img_mask_list()
    print(len(x))