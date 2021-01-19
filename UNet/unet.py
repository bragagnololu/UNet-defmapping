from model_UNET_landslide import *
from data import *
import math
import numpy as np

from sklearn.preprocessing import StandardScaler

import os
import tensorflow as tf

#---------- INPUT VARIABLES ----------------------------------------------------------------------------------------
# loading arrays
image_array = np.load("image_array.npy") # array of training images
image_array[image_array > 10000] = 10000
image_array = image_array.astype(float)/10000
mask_array = np.load("mask_array.npy") # array of training masks

val_image_array = np.load("val_image_array.npy") # array of validation images
val_image_array[val_image_array > 10000] = 10000
val_image_array = val_image_array.astype(float)/10000
val_mask_array = np.load("val_mask_array.npy") # array of validation masks 

channels_imgs = 4 # number of channels of one image

bands_third = np.zeros(channels_imgs)
bands_nin = np.zeros(channels_imgs)

# getting the percentiles of the training array for normalization
for i in range(channels_imgs):
    bands_third[i] = np.percentile(image_array[:,:,:,i],3)
    bands_nin[i] = np.percentile(image_array[:,:,:,i],97)
    
for i in range(channels_imgs):
    image_array[:,:,:,i] = (image_array[:,:,:,i] - bands_third[i])/(bands_nin[i] - bands_third[i])
    val_image_array[:,:,:,i] = (val_image_array[:,:,:,i] - bands_third[i])/(bands_nin[i] - bands_third[i])


nmbr_val = 200 # number of images for validation

batch_size = 1

channels_label = 1 # number of channels of the mask image

target_size = (512,512,channels_imgs) # size of images px x px

logger_file = 'logger_file_path.csv' # .csv file to be created to save the metrics for each epoch
weights_file = 'weights_path.hdf5' # .hdf5 file that will be create to save the best weights

logs_dir = '/logs' # directory to save the logs for TensorBoards application 
#--------------------------------------------------------------------------------------------------------------


data_gen_args = dict(rotation_range=180, #[-180,180]
                    width_shift_range=0.5,
                    height_shift_range=0.5,
                    brightness_range=[-0.2,0.2],
                    #shear_range=0.2,
                    #zoom_range=[0.6,0.8],
                    #zoom_range=[0.4,0.9],
                    #horizontal_flip=True,
                    #vertical_flip = True,
                    fill_mode='nearest'
                    #rescale=1./255
                    )

data_gen_args = dict()

data_gen_val = dict()


myGene = trainGenerator(2,image_array,mask_array,data_gen_args,save_to_dir = False)
myGene_val = valGenerator(1,val_image_array,val_mask_array,data_gen_val,save_to_dir = False)

model = unet(input_size = target_size)


csv_logger = CSVLogger(logger_file, separator=',', append=False)
model_checkpoint = ModelCheckpoint(weights_file, monitor='val_loss',
                                   verbose=1, save_best_only=True,mode='min')
tensorboard = TensorBoard(log_dir=logs_dir,
                             histogram_freq=0, 
                             write_graph=True, write_grads=False, write_images=True,
                             embeddings_freq=0, embeddings_layer_names=None, 
                             embeddings_metadata=None, embeddings_data=None,
                             update_freq='epoch')

earlystop = EarlyStopping(monitor='val_loss',mode='min',patience=20)

#model.load_weights("\BC4.hdf5")

model.fit_generator(myGene,steps_per_epoch=1081,epochs=1000,
                    callbacks=[model_checkpoint,earlystop,csv_logger,tensorboard],
                    validation_data = myGene_val, validation_steps = nmbr_val,
                    ) 
