
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error as skmae


with open('parrot.pkl', 'rb') as f:
    ndvi_lst = pickle.load(f)

pop = pd.read_csv('pop')

pop.drop('Unnamed: 0', axis=1, inplace=True)

pop['Population '] = pop['Population '].str.replace(',','')

pop['Population ']  = pop['Population '].astype('float64')
pop['Population ']  = pop['Population '].astype('float64')
pop = pop.append({'Year ': 1999, 'Population ': 92942}, ignore_index=True)
pop = pop.append({'Year ': 2000, 'Population ':101355}, ignore_index=True)

pop.sort_values('Year ', inplace=True)


for i in ndvi_lst:
    i[1] = int(i[1])


ndvi_no2018 = [i for i in ndvi_lst if i[1] != 2018]


ndvi_2018 = [i for i in ndvi_lst if i[1] == 2018]



for i in ndvi_no2018:
    year = i[1]
    for idx, row in pop.iterrows():
        if row['Year '] == year:
            print(row['Year '], year)
            i.append(row['Population '])



tr_img_data = np.array([i[0] for i in ndvi_no2018]).reshape(-1,197,264,1)
tr_lbl_data = np.array([i[2] for i in ndvi_no2018])



X_train, X_test, y_train, y_test = train_test_split(tr_img_data, tr_lbl_data, test_size =0.10)



# model = Sequential()
# # First Conv
# model.add(Convolution2D(filters = 32,         # I specify 6 filters.
#                         kernel_size = 3,          # means a 3x3 filter
#                         activation = 'relu',
#                      padding='same',
#                  input_shape = (197, 264, 1)# Rectified Linear Unit activation
#                 ))
# model.add(MaxPooling2D(pool_size = 5, padding='same'))

# # Second Conv
# model.add(Conv2D(filters = 50,         # I specify 6 filters.
#                         kernel_size = 3,     # means a 3x3 filter
#                         activation = 'relu',
#                  padding='same'# Rectified Linear Unit activation
#                 ))
# model.add(MaxPooling2D(pool_size= 5, padding='same'))

# #Third conv
# model.add(Conv2D(filters = 80,         # I specify 6 filters.
#                         kernel_size = 3,     # means a 3x3 filter
#                         activation = 'relu',
#                  padding='same'# Rectified Linear Unit activation
#                 ))
# model.add(MaxPooling2D(pool_size = 5, padding='same'))
# model.add(Dropout(0.25))


# model.add(Flatten())
# model.add(Dense(128, activation = 'relu')) 
# model.add(Dropout(0.25))
# model.add(Dense(1, activation = 'relu'))


# In[39]:


model = Sequential()

model.add(Conv2D(40, kernel_size=5, padding="same",input_shape=(197, 264, 1), activation = 'relu'))
model.add(Conv2D(50, kernel_size=5, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(100,activation = 'relu' ))
model.add(Dropout(0.2))

model.add(Dense(1, activation = 'relu'))


# In[40]:


early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')


# In[41]:


model.compile(loss = 'mean_squared_error',
# Categorical cross-entropy is common for unordered discrete predictions.
              optimizer = 'adam',
# Adaptive Moment Estimation, "sophisticated gradient descent"
              metrics = [mean_squared_error])




model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100,  callbacks=[early_stop])



score = model.evaluate(X_test, y_test, verbose=0)
print(score)




test_predict = model.predict(X_test)


print(np.sqrt(mean_squared_error(test_predict, y_test)))




cambridg_2018 = np.array([i[0] for i in ndvi_2018]).reshape(-1,197,264,1)




np.mean(model.predict(cambridg_2018))




model.predict(tr_img_data)


