# import 
import random as random
# set rendom seed 
random.seed(0)

def train_val_split(img_paths, mask_paths, train_ratio=0.8,rendom_seed=0 , shuffle=False):
    """
    Split the image and mask paths into train and val sets.
    """
    # split the data into train and val sets

    if shuffle:
        # use rendom permutation to shuffle the data
        random.seed(rendom_seed)
        random.shuffle(img_paths)
        random.seed(rendom_seed)
        random.shuffle(mask_paths)
    # split the data into train and val sets
    
    train_img_paths = img_paths[:int(len(img_paths)*train_ratio)]
    train_mask_paths = mask_paths[:int(len(mask_paths)*train_ratio)]
    val_img_paths = img_paths[int(len(img_paths)*train_ratio):]
    val_mask_paths = mask_paths[int(len(mask_paths)*train_ratio):]


    return train_img_paths, train_mask_paths, val_img_paths, val_mask_paths

