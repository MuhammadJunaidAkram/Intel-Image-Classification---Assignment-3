# Intel Image Classification - Using Data Augmentation and Transfer Learning

## Purpose
This is a tutorial for classifying images through using transfer learning and data augmentation to train CNN models using Keras.

## Dataset
You can download the training and testing dataset using following line of code.

```python
kaggle datasets download -d puneet6060/intel-image-classification
```

Use the following link to download the prdiction dataset and upload it to your google drive. Then use following code.
https://drive.google.com/file/d/1fJ2hfY-gjvOOtFX9ZPJOYOXGfGVGiTCD/view?usp=sharing

```python
!cp "/content/drive/MyDrive/Test_data.zip" "/content/"
!rm -rf seg_pred
!unzip -q Test_data.zip 
```

## Creating Model
Model being used is VGG19 and its last 3 convolutional layers are kept trainable. The image size is **150x150**.

```python
vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Keeping last 3 layers trainable
vgg.get_layer('block5_conv4').trainable = True
vgg.get_layer('block5_conv3').trainable = True
vgg.get_layer('block5_conv2').trainable = True

layer1 = Flatten()(vgg.output)
layer2 = Dense(64, activation='relu')(layer1)
prediction = Dense(len(folders), activation='softmax')(layer2)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)
```

Setting the optimizer and loss for the model.

```python
from keras.optimizers import SGD
sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)

model.compile(
  loss='categorical_crossentropy',
  optimizer= sgd,
  metrics=['accuracy'])
```

## Data Augmentation

```python
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

pred_datagen = ImageDataGenerator(rescale = 1./255)
```
## Model Architecture

![alt text](https://github.com/MuhammadJunaidAkram/Intel-Image-Classification---Assignment-3/blob/main/images/model_plot.png?raw=true)

## Batch Size
The batch size is kept **64** for this model

## Training Model
Model is trained over **25 epochs**.
```python
r = model.fit_generator(generator=training_set,
                    validation_data=test_set,
                    epochs = 25,
                    steps_per_epoch=len(training_set),
                    validation_steps=len(test_set))
```

## Trained Model Weights
Weights of the trained model can be accessed using the following link.
```python
https://drive.google.com/file/d/1WyoAqmMarADItfaiP77S-STD_AqK-bry/view?usp=sharing
```

## Model Performance
**Accuracy**
The training accuracy of the model is 92.3% and testing accuracy of th model is 90.67%. Accuracy over each epoch can be observed in the following given graph.<br />
![alt text](https://github.com/MuhammadJunaidAkram/Intel-Image-Classification---Assignment-3/blob/main/images/accuracy.PNG?raw=true)

**Loss**
The training loss is 21.04% and validation loss is 26.61%. Loss over each epoch can be observed in the following given graph.<br />
![alt text](https://github.com/MuhammadJunaidAkram/Intel-Image-Classification---Assignment-3/blob/main/images/loss.PNG?raw=true)

**Confusion Matrix**
The confusion matrix of the model is as follows.<br />
![alt text](https://github.com/MuhammadJunaidAkram/Intel-Image-Classification---Assignment-3/blob/main/images/confusion_matrix.PNG?raw=true)
