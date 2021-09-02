# Training STIC model for classifying HCC, ICC and metastasis.
# Python 3.6, tensorflow-gpu 1.12.0, keras 2.2.4
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

from CNN_RNN_data import load_data
from CNN_RNN_model import VGG_GRU_model

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

# Hyperparameter settings in STIC model
save_name = 'HIM_STIC'
type_list=["HCC","ICC","Meta"]
resize=224
categories = len(type_list)
batch_size = 32
epochs = 50
learn_rate = 0.001
earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
red_lr= ReduceLROnPlateau(monitor='val_loss',patience=10,verbose=1,factor=0.8)

# Loading the training data
X_train,Z_train,Y_train = load_data(mode="train",type_list=type_list,resize=resize)

# Using data augmentation to dynamically expand the training data
image_gen_train = ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=.1,
                    height_shift_range=.1,
                    horizontal_flip=False,
                    zoom_range=0.1,
                    validation_split = 0.2
                    )

train_data_gen = image_gen_train.flow(
    (X_train, Z_train), y=Y_train,
    batch_size=batch_size,
    shuffle=True,
    subset='training'
    )

# Using a hold-out validation set containing 20% of the training data to guide the training process
val_data_gen = image_gen_train.flow(
    (X_train, Z_train), y=Y_train,
    batch_size=batch_size,
    shuffle=False,
    subset='validation'
    )

model = VGG_GRU_model(categories)
model.summary()
model.compile(optimizer=Adam(lr=learn_rate),loss='categorical_crossentropy',metrics=['accuracy'])

# STIC model training
model.fit_generator(
	    train_data_gen,
	    epochs=epochs,
	    verbose=2,
	    validation_data=val_data_gen,
	    callbacks=[earlyStopping,red_lr],
	    shuffle=False,
        class_weight='auto'
	)

# Saving trained model weights
model_name = save_name+".h5"
model.save(os.path.join("../model",model_name)) 
print("%s model is saved successfully! \n" %(model_name))