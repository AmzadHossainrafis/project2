from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from utils.sort_img_and_mask_dir import make_img_mask_list
import matplotlib.pyplot as plt

class Dataloader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="uint8")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            #normalize the image
            x[j] = keras.preprocessing.image.img_to_array(img) / 255.0
            
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            
          
        return x, y

# if __name__ == '__main__':
#     x,y=make_img_mask_list()
#     train_ds=Dataloader(batch_size=128, img_size=(256,256), input_img_paths=x, target_img_paths=y)
#     batch=train_ds[0]
#     print(batch[0].shape)
#     print(batch[1].shape)
#     print(batch[0].dtype)
#     print(batch[1].dtype)
#     plt.imshow(batch[0][55])
#     plt.imshow(batch[1][55])


    