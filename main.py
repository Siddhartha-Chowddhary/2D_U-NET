from model import *
from data import *
import tensorflow as tf 

#CORE REF: https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

# model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

model = tf.keras.models.load_model("MODELS/unet_Chest_V2.hdf5")
model.summary()

# TEST_IMG = os.listdir("H:/AAH_Backup_Data/Segmentation/TEST_DATA/TEST/")

# for i in TEST_IMG:
#     Path = f"H:/AAH_Backup_Data/Segmentation/TEST_DATA/TEST/{i}"
#     print(Path)
#     img = plt.imread(Path)
#     # img = io.imread(),as_grey = True)
#     # img = img / 255
#     img = trans.resize(img,(512, 512, 1))
#     pred = np.array([img,])

testGene = testGenerator("H:/AAH_Backup_Data/Segmentation/TEST_DATA/TEST/")
results = model.predict(testGene)
saveResult("H:/AAH_Backup_Data/Segmentation/TEST_DATA/RESULTS/", results)