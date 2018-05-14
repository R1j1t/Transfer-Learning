from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception, preprocess_input
from keras.models import load_model
import keras.callbacks as kcall
import numpy as np
import matplotlib.pyplot as plt

## Parameters
output_classes = 2
learning_rate = 0.0001
img_width, img_height,channel = 299, 299, 3
training_examples = 5216 
batch_size = 30 
epochs = 2
resume_model = False
training_data_dir = './chest_xray/train'
val_data_dir = './chest_xray/val'
test_data_dir = './chest_xray/test'

## Model Defination

if resume_model == False:
  model = Sequential()
  model.add(Xception(weights=None , include_top=False,pooling = 'avg'))
  model.add(Dense(units=output_classes, activation='softmax'))

  model.layers[0].trainable = True 

  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=learning_rate),
                metrics=['accuracy'])
  ## If `weights='imagenet'` doesnt work then do following 2 things
  ##	- Replace `weights='imagenet'` with `weights=None`
  ##	- Uncomment the below line
  model.load_weights('xception_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

else:
  model = load_model('chest_xray.h5')

## For printing the name of the 2 layers
for i, layer in enumerate(model.layers):
        print('Layer: ',i+1,' Name: ', layer.name)

## Image generator function for training and validation
img_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

print('Training set:   ',end='')
train_img_generator = img_generator.flow_from_directory(
training_data_dir,
target_size = (img_width,img_height),
batch_size = batch_size,
class_mode = 'categorical')

print('Validation set: ',end='')
val_img_generator = img_generator.flow_from_directory(
                  val_data_dir,
                  target_size = (img_width,img_height),
                  class_mode = 'categorical')

print('Test set:       ',end='')
## Image generator function for testing
test_img_generator = img_generator.flow_from_directory(
                        test_data_dir,
                        target_size = (img_width,img_height),
                        class_mode = 'categorical',
                        batch_size= batch_size,
			                  shuffle = False)

## Callbacks for model training
early_stop = kcall.EarlyStopping(monitor = 'acc', min_delta=0.0001)
tensorboard =kcall.TensorBoard(log_dir='./tensorboard-logs',write_grads=1,batch_size = batch_size)

class LossHistory(kcall.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        
history = LossHistory()

## Training entire layers
model.fit_generator(train_img_generator,
        steps_per_epoch = training_examples // batch_size,
        epochs = epochs,
        validation_data = val_img_generator,
                validation_steps = 1,
                callbacks=[early_stop,history])

## saving model
model.save('chest_xray.h5')

## Evaluating the model
test_accu = model.evaluate_generator(test_img_generator,steps=624 // batch_size)

## Declaring results
print('Accuracy on test data is:', test_accu[1])
print('Loss on test data is:', test_accu[0])

## Training  Visualisation

### Training loss vs batches trained
plt.plot(history.losses,'b--',label='Training')
plt.plot(len(history.losses)-1,test_accu[0],'go',label = 'Test')

plt.xlabel('# of batches trained')
plt.ylabel('Training loss')

plt.title('Training loss vs batches trained')

plt.legend()

plt.ylim(0,1.2)
plt.show()

### trainng accuracy vs batches trained
plt.plot(history.acc,'--',label= 'Training')
plt.plot(len(history.acc)-1,test_accu[1],'go',label='Test')

plt.xlabel('# of batches trained')
plt.ylabel('Training accuracy')

plt.title('Training accuracy vs batches trained')

plt.legend(loc=4)
plt.ylim(0,1.1)
plt.show()
