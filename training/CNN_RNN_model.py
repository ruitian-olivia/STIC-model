# Define functions for STIC model, Naive RGB model, and Naive joint model.
# Python 3.6, tensorflow-gpu 1.12.0, keras 2.2.4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D,concatenate, Lambda, GRU
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.regularizers import l2
from keras import backend as K

def RGB_slice(x,index):
	"""
	Getting one phase CECT image from one channel of the RGB three-channel image
	Arguments
		x: RGB three-channel images.
		index: the channel index of RGB images {0:Red channel,1:Green channel,2:Blue channel}.
	Returns
		Gray scale images of selected channel. 
	"""
	x_gray = x[:,:,:,index]
	x_gray = K.reshape(x_gray,(-1,224,224,1))
	x_gray = K.concatenate( [x_gray,x_gray,x_gray], axis=-1 )
	
	return x_gray

def CNN_encoder():
	"""
	SpatialExtractor, using VGG-Net pretrained on ImageNet to extract spatial features of images
	Input tensor: (None, 7, 7, 2048)
	Output tensor: (None, 128)
	"""
	inp = Input(shape = (224,224,3), name='Gray_Image_Input')
	vgg_model = VGG16(include_top=False,input_tensor=inp,input_shape=(224,224,3),weights="imagenet")
	for layer in vgg_model.layers:
		layer.trainable = False
	block5_pool = vgg_model.output 
	fc1to_conv = Conv2D(filters=128,kernel_size=(7,7),name='fc1to_conv')(block5_pool)
	encoder_output = Flatten()(fc1to_conv)
	encoder_model = Model(inputs=inp, outputs=encoder_output)

	return encoder_model

def GRU_input(input):
	"""
	Concatenate the three output tensors of SpatialExtractor into the input tensor of TemporalEncoder. 
	Arguments
		input: list of SpatialExtractor output tensors.
	Returns
		Concatenated tensor (None, 3, 128)
	"""
	x1 = input[0]
	x2 = input[1]
	x3 = input[2]
	x1 = K.reshape(x1,(-1,1,128))
	x2 = K.reshape(x2,(-1,1,128))
	x3 = K.reshape(x3,(-1,1,128))
	encoder_concatenated = K.concatenate( [x1,x2,x3], axis=-2 )

	return encoder_concatenated
	
def VGG_GRU_model(category):
	"""
	Define SpatialExtractor-TemporalEncoder-Integration-Classifier(STIC) model.
	Arguments
		category: int, the number of categories the model can classify.
	Returns
		STIC model.
	"""
	# SpatialExtractor module 
	multi_image = Input(shape=(224,224,3))
	x_r = Lambda(RGB_slice,output_shape=(224,224,3),arguments={'index':0})(multi_image)
	x_g = Lambda(RGB_slice,output_shape=(224,224,3),arguments={'index':1})(multi_image)
	x_b = Lambda(RGB_slice,output_shape=(224,224,3),arguments={'index':2})(multi_image)
	encoder_block = CNN_encoder()
	x_r = encoder_block(x_r)
	x_g = encoder_block(x_g)
	x_b = encoder_block(x_b)
	x_encoder= Lambda(GRU_input)([x_r,x_g,x_b])

	# TemporalEncoder module 
	rnn_gru = GRU(32,return_sequences=False)
	x_features = rnn_gru(x_encoder)

	# Integration module 
	structured_data = Input(shape=(20,))
	concatenated = concatenate([x_features,structured_data],axis=-1)

	# Classifier module 	
	predictions = Dense(category, activation= 'softmax')(concatenated)
	model = Model(inputs=[multi_image,structured_data], outputs=predictions)

	return model

def VGG_model(category, l2_rate, clinical_flag):
	"""
	Define Naive RGB/joint model.
	Arguments
		category: int, the number of categories the model can classify.
		l2_rate: float.
		clinical_flag: bool (False: Naive RGB model, True: Naive joint model)
	Returns
		Naive RGB/joint model.
	"""
	multi_image = Input(shape = (224,224,3), name='RGB_Image_Input')
	vgg_model = VGG16(include_top=False,input_tensor=multi_image,input_shape=(224,224,3),weights="imagenet")
	for layer in vgg_model.layers:
		layer.trainable = False
	block5_pool = vgg_model.output
	x = Flatten()(block5_pool)
	x = Dense(512, activation='relu')(x)
	x = Dense(128, activation='relu')(x)
	x = Dense(32, activation='relu', kernel_regularizer=l2(l2_rate))(x)

	# Naive joint model
	if clinical_flag==True:
		structured_data = Input(shape=(20,))
		concatenated = concatenate([x,structured_data],axis=-1)
		predictions = Dense(category, activation= 'softmax')(concatenated)
		model = Model(inputs=[multi_image,structured_data], outputs=predictions)
	# Naive RGB model
	else:
		predictions = Dense(category, activation= 'softmax')(x)
		model = Model(inputs=multi_image, outputs=predictions)
	
	return model