
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from model import *
from data import *


if __name__ == "__main__":
    """ Load the test images """
        # Path = f"H:/AAH_Backup_Data/Segmentation/TEST_DATA/TEST/"

    test_images = os.listdir("H:/AAH_Backup_Data/Segmentation/TEST_DATA/TEST/")

    """ Load the model """

    model = tf.keras.models.load_model("unet_Chest.hdf5")
    model.summary()

    for i in test_images:
        x = cv2.imread(os.path.join("H:/AAH_Backup_Data/Segmentation/TEST_DATA/TEST/", i), cv2.IMREAD_GRAYSCALE)
        original_image = x
        print(x.shape)
        # h, w, _ = x.shape

        x = cv2.resize(x, (512, 512))
        x = x/255.0
        x = x.astype(np.float32)

        x = np.expand_dims(x, axis=0)
        pred_mask = model.predict(x)[0]

        pred_mask = np.concatenate(
            [
                pred_mask,
                pred_mask,
                pred_mask
            ], axis=2)
        pred_mask = (pred_mask > 0.5) * 255
        pred_mask = pred_mask.astype(np.float32)
        pred_mask = cv2.resize(pred_mask, (1024, 1024))

        original_image = original_image.astype(np.float32)
        
        alpha = 0.6
        # cv2.addWeighted(pred_mask, alpha, original_image, 0.7, 0)

        name = i.split("/")[-1]
        cv2.imwrite(f"H:/AAH_Backup_Data/Segmentation/TEST_DATA/RESULTS/{name}", pred_mask)
