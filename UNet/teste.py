from model_UNET_landslide import *
from data import *

import numpy as np
import os

#---------- VARIABLES ----------------------------------------------------------------------------------------
nmbr_imgs_test = 40 # number of test images
target_size = (512,512,4)

bands_third = np.load("bands_third.npy")
bands_nin = np.load("bands_nin.npy")

test_path = 'path_with_test_images'   # inside this directory there must be the 'image', and 'predicted' folders
test_image_folder = 'image'
test_results_folder = 'predicted'
test_probability = 'probabilities'
threshold = 0.5 # which defines what is forest (1) and what isn't (0)

model_weights = "path_with_hdf5_file_saved_on_training"
#--------------------------------------------------------------------------------------------------------------


#testGene = testGenerator("D:/Usuarios/Xeon/Desktop/test_creating_paths/train_unet/test")
testGene = testGenerator2(test_path, test_image_folder,
                          nmbr_imgs_test,target_size,bands_third,bands_nin) # salvar as estatisticas das bandas
model = unet(input_size = target_size)
model.load_weights(model_weights) #mudar aqui
#results = model.predict_generator(testGene,30,verbose=1)
results = model.predict_generator(testGene,nmbr_imgs_test,verbose=1)
print(results.shape)


saveResult3(os.path.join(test_path,test_image_folder),
           os.path.join(test_path,test_results_folder),
           os.path.join(test_path,test_probability),
           results,threshold)