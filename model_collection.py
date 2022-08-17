"""
project: deep learning for infected cell segmentation

author Information
==================
name: amzad hossain rafi
email: amzad.rafi@northsouth.edu
github:

"""





import keras_unet_collection.models as kuc
import yaml
import segmentation_models as sm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
from utils.read_yaml import read_yaml




config = read_yaml(path='traning.yaml')

# Segmentation Models unet/linknet/fpn/pspnet
def get_model(model_name='unet'):
    """
    arguments: no arguments
    
    return: unet model
    
    """
    if model_name == 'unet':
        model = sm.Unet(backbone_name='efficientnetb0', input_shape=(config['height'], config['width'], config['in_channels']),
                        classes = config['num_classes'], activation='softmax',
                        encoder_weights=None, weights=None)
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
     


    if model_name == 'link_net':
        model = sm.Linknet(backbone_name='efficientnetb0', input_shape=(config['height'], config['width'], config['in_channels']),
                        classes = config['num_classes'], activation='softmax',
                        encoder_weights=None, weights=None)
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
        

    if model_name == 'FPN' :

        model = sm.FPN(backbone_name='efficientnetb0', input_shape=(config['height'], config['width'], config['in_channels']),
                        classes = config['num_classes'], activation='softmax',
                        encoder_weights=None, weights=None)
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
       



    if model_name == "PSP_net":

        model = sm.PSPNet(backbone_name='efficientnetb0', input_shape=(config['height'], config['width'], config['in_channels']),
                        classes = config['num_classes'], activation='softmax',
                        encoder_weights=None, weights=None)
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
     



    if model_name=='vnet':

        model = kuc.vnet_2d((config['height'], config['width'], config['in_channels']), filter_num=[16, 32, 64, 128, 256], 
                            n_labels=config['num_classes'] ,res_num_ini=1, res_num_max=3, 
                            activation='PReLU', output_activation='Softmax', 
                            batch_norm=True, pool=False, unpool=False, name='vnet')
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
      



    if model_name=="unet3pp":

        model = kuc.unet_3plus_2d((config['height'], config['width'], config['in_channels']), 
                                    n_labels=config['num_classes'], filter_num_down=[64, 128, 256, 512], 
                                    filter_num_skip='auto', filter_num_aggregate='auto', 
                                    stack_num_down=2, stack_num_up=1, activation='ReLU', output_activation='Softmax',
                                    batch_norm=True, pool='max', unpool=False, deep_supervision=True, name='unet3plus')
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
      




    if model_name=='r2net':

        model = kuc.r2_unet_2d((config['height'], config['width'], config['in_channels']), [64, 128, 256, 512], 
                                n_labels=config['num_classes'],
                                stack_num_down=2, stack_num_up=1, recur_num=2,
                                activation='ReLU', output_activation='Softmax', 
                                batch_norm=True, pool='max', unpool='nearest', name='r2unet')
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)




    if model_name=='unetp2d':

        model = kuc.unet_plus_2d((config['height'], config['width'], config['in_channels']), [64, 128, 256, 512], 
                                n_labels=config['num_classes'],
                                stack_num_down=2, stack_num_up=1, recur_num=2,
                                activation='ReLU', output_activation='Softmax', 
                                batch_norm=True, pool='max', unpool='nearest', name='r2unet')
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)






    if model_name=='resnet_Unet':

        model = kuc.resunet_a_2d((config['height'], config['width'], config['in_channels']), [32, 64, 128, 256, 512, 1024], 
                                dilation_num=[1, 3, 15, 31], 
                                n_labels=config['num_classes'], aspp_num_down=256, aspp_num_up=128, 
                                activation='ReLU', output_activation='Softmax', 
                                batch_norm=True, pool=False, unpool='nearest', name='resunet')
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
     





    if model_name=='transnet':
        model = kuc.transunet_2d((config['height'], config['width'], config['in_channels']), filter_num=[64, 128, 256, 512],
                                n_labels=config['num_classes'], stack_num_down=2, stack_num_up=2,
                                embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
                                activation='ReLU', mlp_activation='GELU', output_activation='Softmax', 
                                batch_norm=True, pool=True, unpool='bilinear', name='transunet')
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
       





    if model_name=="swin":
        model = kuc.swin_unet_2d((config['height'], config['width'], config['in_channels']), filter_num_begin=64, 
                                n_labels=config['num_classes'], depth=4, stack_num_down=2, stack_num_up=2, 
                                patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
                                output_activation='Softmax', shift_window=True, name='swin_unet')
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
       




    if model_name=='u2net':
        model = kuc.u2net_2d((config['height'], config['width'], config['in_channels']), n_labels=config['num_classes'], 
                                filter_num_down=[64, 128, 256, 512], filter_num_up=[64, 64, 128, 256], 
                                filter_mid_num_down=[32, 32, 64, 128], filter_mid_num_up=[16, 32, 64, 128], 
                                filter_4f_num=[512, 512], filter_4f_mid_num=[256, 256], 
                                activation='ReLU', output_activation='Softmax', 
                                batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net')
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)
        






    if model_name=='att_unet':
  
        model = kuc.att_unet_2d((config['height'], config['width'], config['in_channels']), [64, 128, 256, 512], 
                                n_labels=config['num_classes'],
                                stack_num_down=2, stack_num_up=2,
                                activation='ReLU', atten_activation='ReLU', attention='add', output_activation=None, 
                                batch_norm=True, pool=False, unpool='bilinear', name='attunet')
        x = model.layers[-2].output # fetch the last layer previous layer output
        
        output = Conv2D(config['num_classes'], kernel_size = (1,1), name="out", activation = 'softmax',dtype="float32")(x) # create new last layer
        model = Model(inputs = model.input, outputs=output)

    return model
