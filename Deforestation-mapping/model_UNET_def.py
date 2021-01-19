import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, EarlyStopping, TensorBoard, Callback
from keras import backend as keras
import tensorflow as tf
from bce_loss import *
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def unet(pretrained_weights = None,input_size = (512,512,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)


    model = Model(input = inputs, output = conv10)
    
    # def binary_focal_loss(gamma, alpha):
    #     """
    #     Binary form of focal loss.
    #       FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    #       where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    #     References:
    #         https://arxiv.org/pdf/1708.02002.pdf
    #     Usage:
    #      model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    #     """
    #     def binary_focal_loss_fixed(y_true, y_pred):
    #         """
    #         :param y_true: A tensor of the same shape as `y_pred`
    #         :param y_pred:  A tensor resulting from a sigmoid
    #         :return: Output tensor.
    #         """
                
    #         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    #         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    #         epsilon = keras.epsilon()
    #         # clip to prevent NaN's and Inf's
    #         pt_1 = keras.clip(pt_1, epsilon, 1. - epsilon)
    #         pt_0 = keras.clip(pt_0, epsilon, 1. - epsilon)

    #         return -keras.sum(alpha * keras.pow(1. - pt_1, gamma) * keras.log(pt_1))-keras.sum((1-alpha) * keras.pow( pt_0, gamma) * keras.log(1. - pt_0))


    #     return binary_focal_loss_fixed
    
    
    # WITH THRESHOLD
    def precision_threshold(threshold=0.3):
        def precision(y_true, y_pred):
            """Precision metric.
            Computes the precision over the whole batch using threshold_value.
            """
            threshold_value = threshold
            # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
            y_pred = keras.cast(keras.greater(keras.clip(y_pred, 0, 1), threshold_value), keras.floatx())
            # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
            true_positives = keras.round(keras.sum(keras.clip(y_true * y_pred, 0, 1)))
            # count the predicted positives
            predicted_positives = keras.sum(y_pred)
            # Get the precision ratio
            precision_ratio = true_positives / (predicted_positives + keras.epsilon())
            return precision_ratio
        return precision
    
    def recall_threshold(threshold = 0.3):
        def recall(y_true, y_pred):
            """Recall metric.
            Computes the recall over the whole batch using threshold_value.
            """
            threshold_value = threshold
            # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
            y_pred = keras.cast(keras.greater(keras.clip(y_pred, 0, 1), threshold_value), keras.floatx())
            # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
            true_positives = keras.round(keras.sum(keras.clip(y_true * y_pred, 0, 1)))
            # Compute the number of positive targets.
            possible_positives = keras.sum(keras.clip(y_true, 0, 1))
            recall_ratio = true_positives / (possible_positives + keras.epsilon())
            return recall_ratio
        return recall
        
    
    #F1 metric
    def recall_m(y_true, y_pred):   
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = keras.cast(keras.greater(keras.clip(y_pred, 0, 1), 0.3), keras.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = keras.round(keras.sum(keras.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = keras.sum(keras.clip(y_true, 0, 1))
        recall = true_positives / (possible_positives + keras.epsilon())
        return recall

    def precision_m(y_true, y_pred):      
        y_pred = keras.cast(keras.greater(keras.clip(y_pred, 0, 1), 0.3), keras.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = keras.round(keras.sum(keras.clip(y_true * y_pred, 0, 1)))
        # Count the predicted positives
        predicted_positives = keras.sum(y_pred)
        # Get the precision ratio
        precision = true_positives / (predicted_positives + keras.epsilon())
    
        return precision

    def f1_m(y_true, y_pred):
        precision_value = recall_m(y_true, y_pred)
        recall_value = precision_m(y_true, y_pred)
        return 2*((precision_value*recall_value)/(precision_value+recall_value+keras.epsilon()))
    

    sgd = SGD(lr=0.0001, momentum=0.9, nesterov=True)

    model.compile(optimizer = sgd, loss=[weighted_bce_dice_loss], metrics = [f1_m,recall_threshold(0.3),precision_threshold(0.3)])


    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


