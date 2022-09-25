# ref_1 : https://github.com/zhixuhao/unet
# ref_2 : https://pallawi-ds.medium.com/semantic-segmentation-with-u-net-train-and-test-on-your-custom-data-in-keras-39e4f972ec89
# Ref_3: https://answers.opencv.org/question/225971/how-do-i-use-this-image-as-a-mask-to-perform-segmentation/


# CORE_REF_1 : https://automaticaddison.com/how-to-apply-a-mask-to-an-image-using-opencv/
# CORE_REF_2 : https://www.geeksforgeeks.org/opencv-invert-mask/

from model import *
from data import *
import matplotlib.pyplot as plt

data_gen_args = dict( featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0,
                 width_shift_range=0.05,
                 height_shift_range=0.05,
                 shear_range=0.2,
                 zoom_range=0.2,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=1.0/255)
                 
myGene = trainGenerator(2,'F:/Medical_AI/SEGMENTATION/DATA_2/TRAINING_DATA/TRAIN','IMAGE','MASK', data_gen_args, 
save_to_dir = "F:/Medical_AI/SEGMENTATION/unet-master/AUG_DATA/IMG")


# imgs, labels = next(myGene)

model = unet()
model_checkpoint = ModelCheckpoint('unet_Chest.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=2000,epochs=5,callbacks=[model_checkpoint])



