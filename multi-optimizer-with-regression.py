#multi-optimizers 
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation

def train_given_optimizer(optimiser):
    model = Sequential()
    model.add(Dense(1,input_dim=500))
    model.add(Activation(activition='sigmoid'))
    model.compile(optimizer=optimiser,loss='binary_crossentropy',metrics=['accuracy'])

    data = np.random.random((1000,500))
    labels = np.random.randint(2,size=(1000,1))

    score = model.evaluate(data,labels,verbose=0)
    print("Optimizer:  ",optimiser)
    print("Before training:  ",model.metrics_names,score)

    model.fit(data,labels,epochs=10,batch_size=32,verbose=0)
    score = model.evaluate(data,labels,verbose)

    print("After training:  ",model.metrics_names,score)

train_given_optimizer("sgd")
train_given_optimizer("rmsprop")
train_given_optimizer("adagrad")
train_given_optimizer("adadelta")
train_given_optimizer("adam")
train_given_optimizer("adamax")
train_given_optimizer("nadam")

 