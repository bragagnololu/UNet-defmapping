#load images and use the metrics

import numpy as np
from sklearn.metrics import fbeta_score, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, f1_score
import rasterio
import os

y_true = list()
y_pred = list()

pred_png = list()


'''
--test_images folder
  |__mask folder
  |__predicted images folder
'''

test_path = '/test_images'
true = 'mask'
pred = 'predicted_0.5'

names_images = sorted(os.listdir(os.path.join(test_path,true)))
pred_png += [file.replace(".tif",".png") for file in names_images]


for image in names_images:

    path = os.path.join(test_path,true,image)
    
    img_open = rasterio.open(path)
    np_im = img_open.read()
    
   
    y_true.append(np_im)
    

for image in pred_png:

    path = os.path.join(test_path,pred,image)
    
    img_open = rasterio.open(path)
    np_im = img_open.read()
    

    binarr = np.where(np_im>0, 1, 0)
    
    y_pred.append(binarr)
  

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

y_true = y_true.ravel()
y_pred = y_pred.ravel()

f1_score_teste = f1_score(y_true, y_pred,average='macro')
print("F1-score: ",f1_score_teste)

accuracy_score_teste = accuracy_score(y_true, y_pred)
print("Accuracy: ",accuracy_score_teste)

all_metrics = precision_recall_fscore_support(y_true, y_pred,average="macro")
print("Precision, recall and fscore: ", all_metrics)

#tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#print("TN, FP, FN, TP: ",tn, fp, fn, tp)

cf = confusion_matrix(y_true, y_pred)
print("Confusion matrix: ")
print(cf)


fbeta = fbeta_score(y_true, y_pred, average='macro', beta=0.01)
print("Fbeta: ", fbeta)

print(classification_report(y_true, y_pred))



    
