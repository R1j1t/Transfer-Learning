from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
import keras.callbacks as kcall
import numpy as np
import matplotlib.pyplot as plt

class LossHistory(kcall.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

class chest_xray:

	def __init__ (self):

		## Parameters
		self.output_classes = 2
		self.learning_rate = 0.0001
		self.img_width, self.img_height,self.channel = 299, 299, 3
		self.training_examples = 5216
		self.batch_size = 30
		self.epochs = 2
		self.resume_model = False
		self.training_data_dir = './chest_xray/train'
		self.val_data_dir = './chest_xray/val'
		self.test_data_dir = './chest_xray/test'
		self.img_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

	def model_defination(self):
		## Model Defination
		if self.resume_model == False:

			self.model = Sequential()
			self.model.add(Xception(weights=None , include_top=False,pooling = 'avg'))
			self.model.add(Dense(units=self.output_classes, activation='softmax'))

			self.model.layers[0].trainable = True

			self.model.compile(loss='categorical_crossentropy',
			            optimizer=Adam(lr=self.learning_rate),
			            metrics=['accuracy'])
			## If `weights='imagenet'` doesnt work then do following 2 things
			##	- Replace `weights='imagenet'` with `weights=None`
			##	- Uncomment the below line
			self.model.load_weights('xception_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

		else:
			self.model = load_model('chest_xray.h5')

		for i, layer in enumerate(self.model.layers):
			print('Layer: ',i+1,' Name: ', layer.name)

	def img_pipeline(self):
		self.train_img_generator = self.img_generator.flow_from_directory(
								self.training_data_dir,
								target_size = (self.img_width,self.img_height),
								batch_size = self.batch_size,
								class_mode = 'categorical')

		self.val_img_generator = self.img_generator.flow_from_directory(
								self.val_data_dir,
								target_size = (self.img_width,self.img_height),
								class_mode = 'categorical')

		self.test_img_generator = self.img_generator.flow_from_directory(
		                        self.test_data_dir,
		                        target_size = (self.img_width,self.img_height),
		                        class_mode = 'categorical',
		                        batch_size= self.batch_size,
					            shuffle = False)

		return ('Image generating pipelines loaded successfully')

	def callbacks(self):

		## Callbacks for model training
		self.early_stop = kcall.EarlyStopping(monitor = 'acc', min_delta=0.0001)
		self.tensorboard =kcall.TensorBoard(log_dir='./tensorboard-logs',write_grads=1,batch_size = self.batch_size)
		self.history = LossHistory()

	def training(self):
		self.model.fit_generator(self.train_img_generator,
	        steps_per_epoch = self.training_examples // self.batch_size,
	        epochs = self.epochs,
	        validation_data = self.val_img_generator,
			validation_steps = 1,
			callbacks=[self.early_stop,self.history])

		## saving model
		self.model.save('chest_xray.h5')

	def evaluation(self):
		test_accu = self.model.evaluate_generator(self.test_img_generator,steps=624 // self.batch_size)

		## Declaring results
		print('Accuracy on test data is:', test_accu[1])
		print('Loss on test data is:', test_accu[0])

chest = chest_xray()
chest.callbacks()
chest.img_pipeline()
chest.model_defination()
chest.training()
chest.evaluation()

# history = LossHistory()
#
# ## Training entire layers
# if resume_model:
# 	model = load_model('chest_xray.h5')
# else:
#
# ## Image generator function for testing
#
# ## Evaluating the model
# test_accu = model.evaluate_generator(test_img_generator,steps=624 // batch_size)
#
# ## Declaring results
# print('Accuracy on test data is:', test_accu[1])
# print('Loss on test data is:', test_accu[0])
#
# ## Training  Visualisation
#
# ### Training loss vs batches trained
# plt.plot(history.losses,'b--',label='Training')
# plt.plot(len(history.losses)-1,test_accu[0],'go',label = 'Test')
#
# plt.xlabel('# of batches trained')
# plt.ylabel('Training loss')
#
# plt.title('Training loss vs batches trained')
#
# plt.legend()
#
# plt.ylim(0,1.2)
# plt.show()
#
# ### trainng accuracy vs batches trained
#
# plt.plot(history.acc,'--',label= 'Training')
# plt.plot(len(history.acc)-1,test_accu[1],'go',label='Test')
#
# plt.xlabel('# of batches trained')
# plt.ylabel('Training accuracy')
#
# plt.title('Training accuracy vs batches trained')
#
# plt.legend(loc=4)
# plt.ylim(0,1.1)
# plt.show()
