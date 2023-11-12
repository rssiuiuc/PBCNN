import pandas as pd
import numpy as np
import pdb
import pandas as pd
import numpy as np
import pdb
import os
import tensorflow as tf
print(tf.test.gpu_device_name())
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
#import kerastuner as kt
import tensorflow_probability as tfp
dist = tfp.distributions

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from keras.layers import Input

import tensorflow_probability as tfp
dist = tfp.distributions

# define the gaussion distribution for the output
tfd = tfp.distributions
def normal_sp(params):
  return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2]))# both parameters are learnable

# define the loss function,which stands for Negative Log-Likelihood. This function takes two arguments: `y` and `distr`.
'''
The `y` argument represents the observed data. This could be any form of data that you're trying to model, such as the heights of a group of people, the scores of a group of students, etc.

The `distr` argument represents a probability distribution. In the context of this function, it's expected that `distr` is an object that has a method called `log_prob`. This method should take a data point (like `y`) and return the natural logarithm of the probability of that data point according to the distribution.

The function `NLL` calculates the negative log-likelihood of the observed data `y` given the probability distribution `distr`. In other words, it calculates how likely the observed data is if we assume that it follows the given distribution. 

The negative sign is used because likelihoods are usually maximized, while in optimization problems (like training a machine learning model), we usually minimize. So, by taking the negative log-likelihood, we can minimize this value to get the best parameters for our model.

A lower negative log-likelihood means that the model fits the data better.
'''
def NLL(y, distr):
  return -distr.log_prob(y)

#load data
start_year=2008

# all variables except soil (samples,11,6,16)
train_phen=np.load('/content/drive/MyDrive/Yield_paper_results3/Input/train_phen'+str(start_year)+'.npy')
test_phen=np.load('/content/drive/MyDrive/Yield_paper_results3/Input/test_phen'+str(start_year)+'.npy')

# soil variables (samples,11,6,36)
train_soil=np.load('/content/drive/MyDrive/Yield_paper_results3/Input/train_soil'+str(start_year)+'.npy')
test_soil=np.load('/content/drive/MyDrive/Yield_paper_results3/Input/test_soil'+str(start_year)+'.npy')

train_y=np.load('/content/drive/MyDrive/Yield_paper_results3/Input/train_y'+str(start_year)+'.npy')
test_y=np.load('/content/drive/MyDrive/Yield_paper_results3/Input/test_y'+str(start_year)+'.npy')
# Seed value, both for numpy and tensorflow
seed_value= 2022
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

keras.backend.clear_session()

# define the model, MC-dropout is activated by setting training=True
# set the input shape
input_all = keras.Input(shape=(11, 6, 4+5+2+5))
inputA = layers.Lambda(lambda x: x[:,:,:,0:4])(input_all)# 0:4 means EVI2 group
inputB = layers.Lambda(lambda x: x[:,:,:,4:4+4])(input_all)# 4:4+4 means the Heat-related group
inputC = layers.Lambda(lambda x: x[:,:,:,8:])(input_all) # 8: means the Water-related group
input_soil=keras.Input(shape=(11, 6, 6*6))# soil group

x1=layers.Conv2D(4, kernel_size=(1,1), strides=(1,1),activation=tf.nn.relu)(inputA)
x1=layers.BatchNormalization()(x1)
x1=layers.Dropout(0.25)(x1, training=True) # training=True means the dropout is activated during both training and testing time

x2=layers.Conv2D(4, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu)(inputB)
x2=layers.BatchNormalization()(x2)
x2=layers.Dropout(0.5)(x2, training=True)

x3=layers.Conv2D(8, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu)(inputC)
x3=layers.BatchNormalization()(x3)
x3=layers.Dropout(0.5)(x3, training=True)

x4=layers.Conv2D(8, kernel_size=(1,1), strides=(1,1), activation=tf.nn.relu)(input_soil)
x4=layers.BatchNormalization()(x4)
x4=layers.Dropout(0.5)(x4, training=True)

merged = layers.Concatenate()([x1,x2,x3,x4])
z=layers.Dropout(0.5)(merged, training=True)

z=layers.Conv2D(32, kernel_size=(3,3), strides=(1,1),padding='same',
                            activation=tf.nn.relu)(z)
z=layers.BatchNormalization()(z)
z=layers.Dropout(0.25)(z, training=True)

z=layers.Conv2D(16, kernel_size=(3,3), strides=(1,1),padding='same',
activation=tf.nn.relu)(z)
z=layers.BatchNormalization()(z)
z=layers.Dropout(0.25)(z, training=True)

z=layers.Flatten()(z)

z=layers.Dense(1024)(z)
z=layers.Dropout(0.1)(z, training=True)

params = layers.Dense(2)(z)
outputs = tfp.layers.DistributionLambda(normal_sp)(params)# the output is a distribution

model=tf.keras.Model(inputs=[input_all,input_soil], outputs=outputs)

# compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss=NLL,metrics=['RootMeanSquaredError']) # the loss function is NLL

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# fit model
history = model.fit([train_phen,train_soil], train_y, epochs=1000, batch_size=32, verbose=1, validation_split = 0.2, callbacks=[es])

# retrain the model with a smaller learning rate
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt,loss=NLL,metrics=['RootMeanSquaredError'])

# fit model
history = model.fit([train_phen,train_soil], train_y, epochs=300, batch_size=32, verbose=1, validation_split = 0.2, callbacks=[es])

# evaluate the model by MC-dropout
test_input=[test_phen,test_soil]
# predict the mean and std of the output
predicted = []
predicted_stds = []
predicted_means = []

for _ in range(10000):
    predicted.append(model.predict([test_input]))

    prediction_distribution = model(test_input)
    prediction_mean = prediction_distribution.mean().numpy().tolist() # the mean of the output
    prediction_stdv = prediction_distribution.stddev().numpy() # the std of the output

    predicted_means.append(prediction_mean)
    predicted_stds.append(prediction_stdv)

mu_pred =np.mean(predicted,axis=0)
print('rmse',np.sqrt(sum((mu_pred.reshape([test_y.shape[0],])-test_y)**2)/test_y.shape[0]))

sigma_pred =np.std(predicted,axis=0) # the total uncertainty
print('averaged total uncertainty',np.mean(sigma_pred))

sigma_model =np.std(predicted_means,axis=0) # the epistemic uncertainty
sigma_data =np.mean(predicted_stds,axis=0) # the aleatoric uncertainty

print('averaged epistemic uncertainty',np.mean(sigma_model))
print('averaged aleatoric uncertainty',np.mean(sigma_data))

