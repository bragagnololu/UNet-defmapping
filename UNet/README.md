*UNet-defmapping/UNet* has the codes used to carry out UNet training, considering the problem of binary forest/non-forest classification in images from the Sentinel-2 satellite.

**Step 1:** *gen_npy_files.py* -> Transform training/validation images and masks into arrays.

Required folder structure: 
<br />
```
--Training (define in lines 19-21)
 |__image
 |__label

--Validation (define in lines 67-69)
 |__image
 |__label
```
*Set image size on line 22.


**Step 2:**
*unet.py* -> UNet training. Parameter inputs must be made on lines 11-49.
Attention: at the end of the training, save the 'bands_third.npy' and 'bands_nin.npy' arrays if the deforestation monitoring system will be used.


**Step 3:**
*teste.py* -> After training, you can apply test images for evaluation. Change of variables and parameters in lines 8-20.

Required folder structure:

    --Test
     |__images
     |__predictions (empty)
     |__probabilities (empty)

**Step 4:**
*metrics_calc_test.py* -> Calculates metrics for UNet classified images and reference images (masks).


