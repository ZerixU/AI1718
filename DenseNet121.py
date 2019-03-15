from keras import Model
from keras.layers import  Dense,GlobalAveragePooling2D
nb_classes=8
from keras.applications.densenet import DenseNet121
FirstModel=DenseNet121( weights= 'imagenet', include_top=False)
x=FirstModel.output
x=GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction= Dense(nb_classes, activation='softmax') (x)
NewModel=Model(inputs=FirstModel.input, outputs=prediction)
for layer in FirstModel.layers:
    layer.trainable = False
NewModel.compile(optimizer='rmsprop', loss='categorical_crossentropy')

from keras.preprocessing.image import ImageDataGenerator
train_dataGenerator=ImageDataGenerator(rescale=1./255)
test_dataGenerator=ImageDataGenerator(rescale=1./255)
train_gen=train_dataGenerator.flow_from_directory(directory="KaggleTrain", batch_size=16,target_size=(299, 299))
test_gen=test_dataGenerator.flow_from_directory(directory="KaggleTest", batch_size=16,target_size=(299, 299))
NewModel.fit_generator(train_gen,epochs=4, validation_data=test_gen,validation_steps=1727//16,steps_per_epoch=5172//16)

from keras.preprocessing.image import ImageDataGenerator
for layers in NewModel.layers[:-3]:
    layers.trainable=False

from keras.optimizers import SGD
NewModel.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])

structure= NewModel.to_json()
with open("DensNet121-kaggle-savedStructured", "w") as sheila:
    sheila.write(structure)
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint= ModelCheckpoint('DensNet121-kaggle-BestNetwork.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stopper = EarlyStopping(min_delta=0.01, patience=2)
train_dataGenerator=ImageDataGenerator(rescale=1./255)
test_dataGenerator=ImageDataGenerator(rescale=1./255)
train_gen=train_dataGenerator.flow_from_directory(directory="KaggleTrain", batch_size=16,target_size=(224, 224))
test_gen=test_dataGenerator.flow_from_directory(directory="KaggleTest", batch_size=16,target_size=(224, 224))
NewModel.fit_generator(train_gen,epochs=20, validation_data=test_gen,validation_steps=1727//16,steps_per_epoch=5172//16,callbacks=[checkpoint,early_stopper])