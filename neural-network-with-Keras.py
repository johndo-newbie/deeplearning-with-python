#single layer neural network with Keras
import numpy as numpy
from keras.model import Sequenrial
from keras.layer import Dense,Activation
from keras.utils.visualize_util import plot

model = Sequential()
model.add(Dense(1,input_dim = 500))
model.add(Activation(activation='sigmoid'))
model.compile(optimizer = 'rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

data = np.random.random((1000,500))
labels = np.random.randint(2,size=(1000,1))

score = model.evaluate(data,labels,verbose = 0)
print("Before Training: ",zip(model.metrics_names,score))

model.fit(data,labels,nb_epoch=10,batch_size = 32,verbose=0)

score = model.evaluate(data,labels,verbose=0)
print("After training: ",zip(model.metrics_names,score))
plot(model,to_file='s1.png',show_shapes=True)
