from utils.read_yaml import  read_yaml
import segmentation_models as sm
import yaml 

from model_collection import get_model,config
from tensorflow.keras import mixed_precision
from utils.sort_img_and_mask_dir import make_img_mask_list
from dataloader import Dataloader


def train(config):
    # Equivalent to the two lines above
    mixed_precision.set_global_policy('mixed_float16')
    #config = read_yaml(path="traning.yaml")

    model=get_model(config["model_name"])
    model.compile(optimizer=config["optimizer"], loss=config['loss'], metrics=['accuracy'])
    # model.summary()
    x,y=make_img_mask_list()

    #split x, y  data into train img ,train mask and val img,val mask
    train_img = x[:int(len(x)*0.8)]
    train_mask = y[:int(len(y)*0.8)]
    val_img = x[int(len(x)*0.8):]
    val_mask = y[int(len(y)*0.8):]


    train_ds=Dataloader(batch_size=config["batch_size"], img_size=(config["height"],config["width"]), input_img_paths=train_img, target_img_paths=train_mask)
    val_ds= Dataloader(batch_size=config["batch_size"], img_size=(config["height"],config["width"]), input_img_paths=val_img, target_img_paths=val_mask)
    model.fit(train_ds, epochs=config["epochs"], validation_data=val_ds, verbose=1)




if __name__ == "__main__":
    train(config)