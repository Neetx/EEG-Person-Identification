import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pyedflib
from tqdm import tqdm
import time

def CNN():
    input_shape = (160, 64)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape, epsilon=.0001))

    model.add(tf.keras.layers.Conv1D(input_shape=input_shape, activation='relu', filters=128, kernel_size=2, strides=1, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(tf.keras.layers.Conv1D(input_shape=(80, 128), activation='relu', filters=256, kernel_size=2, strides=1, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(tf.keras.layers.Conv1D(input_shape=(40, 256), activation='relu', filters=512, kernel_size=2, strides=1, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(tf.keras.layers.Conv1D(input_shape=(20, 512), activation='relu', filters=1024, kernel_size=2, strides=1, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(tf.keras.layers.Reshape((-1, 64*160), input_shape=(80, 10, 1024)))
    #model.add(layers.Reshape((-1, 64*160)))
    model.add(tf.keras.layers.Dropout(rate=0.5, input_shape=(80, 10240))) #0.5 is keep_prob
    #model.add(layers.Dropout(0.5)) #0.5 is keep_prob
    
    model.add(tf.keras.layers.Dense(109, activation='softmax'))

    model.summary()
    return model

def _read_py_function(filename):
    #print("DEBUGGGG")
    #print(filename.numpy().decode())
    f = pyedflib.EdfReader(filename.numpy().decode())
    n_channels = f.signals_in_file
    channels = f.getSignalLabels()
    eeg_data = np.zeros((n_channels, f.getNSamples()[0]), dtype=np.float32)
    for i in np.arange(n_channels):
        eeg_data[i, :] = f.readSignal(i)

    n_samples = f.getNSamples()[0]
    reminder = int(n_samples % 160)
    n_samples -= reminder
    #seconds = int((n_samples - (n_samples % 160))/160) #160 is frequency
    seconds = int(n_samples/160) #160 is frequency
    
    path = filename.numpy().decode().split("\\")
    person_id = int(path[-1].partition("S")[2].partition("R")[0]) #extract number between S and R
    label = np.zeros(109, dtype=bool) #109 classes (persons)
    label[person_id-1]=1
    labels = np.tile(label, (seconds,1))
    
    eeg_data = eeg_data.transpose()
    if reminder > 0:
        eeg_data = eeg_data[:-reminder, :]
    intervals = np.linspace(0, n_samples, num=seconds, endpoint=False, dtype=int)
    eeg_data = np.split(eeg_data, intervals) #return a list, remove the first empty 
    del eeg_data[0]
    eeg_data = np.array(eeg_data)   #shape = (seconds, frequency, n_channels)
    #eeg_data = _batch_normalization(eeg_data)
    
    """print(n_channels)
    print(n_samples)
    print(seconds)
    print(eeg_data)
    print(eeg_data.shape)"""
    #print(labels)
    #print(labels.shape)
    
    # eeg_data = np.transpose(eeg_data, (1, 0))
    # eeg_data = scale(eeg_data, with_std=False, axis=1)

    return eeg_data, labels#, trial_num[0]

def get_dataset2(input="train"):
    path = "C:\\PATH\\TO\\files\\"
    if input=="train":
        dataset = tf.data.Dataset.list_files(path + "S*\S*R01.edf")  
        for i in range(2, 13):
            nth_record = tf.data.Dataset.list_files(path + "S*\S*R" + "{:02d}".format(i) + ".edf")
            dataset = dataset.concatenate(nth_record)
    elif input=="test":
        dataset = tf.data.Dataset.list_files(path + "S*\S*R13.edf")  
    elif input=="validation":
        dataset = tf.data.Dataset.list_files(path + "S*\S*R14.edf")  


    length = len(list(dataset.as_numpy_iterator()))
    train_data = list()   #List.append() instead of np.append()
    labels = list()
    
    #index = 0
    with tqdm(total=length) as pbar:
        for filename in dataset:
            eeg_data, label = _read_py_function(filename)
            train_data.append(eeg_data)
            label = np.expand_dims(label, axis=1)
            labels.append(label)
            #index += 1
            #if index == 10:
            #  break
            pbar.update(1)
    print("Loaded")
    
    return train_data, labels

training_dataset, training_labels = get_dataset2(input="train")

train_data = np.empty([1, 160, 64], dtype=np.float32)
train_data = np.vstack(training_dataset)
del(training_dataset)

train_label = np.empty([1, 1, 109], dtype=bool)
train_label = np.vstack(training_labels)
del(training_labels)

testing_dataset, testing_labels = get_dataset2(input="test")

test_data = np.empty([1, 160, 64], dtype=np.float32)
test_data = np.vstack(testing_dataset)
del(testing_dataset)

test_label = np.empty([1, 1, 109], dtype=bool)
test_label = np.vstack(testing_labels)
del(testing_labels)

val_dataset, val_labels = get_dataset2(input="validation")

val_data = np.empty([1, 160, 64], dtype=np.float32)
val_data = np.vstack(val_dataset)
del(val_dataset)

val_label = np.empty([1, 1, 109], dtype=bool)
val_label = np.vstack(val_labels)
del(val_labels)

model = CNN()

tf.keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer='adam',
          loss=tf.keras.losses.CategoricalCrossentropy(),
          metrics=['accuracy'])

print(train_label.shape)
print(test_label.shape)
print(val_label.shape)

checkpoint_path = "C:\\PATH\\TO\\checkpoint_cnn_00001_12-1-1\\cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model.save_weights(checkpoint_path.format(epoch=0))

with tf.device('/CPU:0'):
    history = model.fit(train_data, train_label, epochs=150, validation_data=(test_data, test_label), batch_size = 80,  callbacks=[cp_callback])

np.save("C:\\PATH\\TO\\checkpoint_cnn_00001_12-1-1\\history.npy", history.history)
history = np.load("C:\\PATH\\TO\\checkpoint_cnn_00001_12-1-1\\history.npy", allow_pickle='TRUE').item()

if type(history) is not dict:
    history = history.history

plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

max_value = max(history['val_accuracy'])
print(max_value)
max_index = history['val_accuracy'].index(max_value)
print(max_index)
print(history['accuracy'][max_index])
best_checkpoint_path = "C:\\PATH\\TO\\checkpoint_cnn_00001_12-1-1\\cp-{:04d}.ckpt".format(max_index)
model.load_weights(best_checkpoint_path)

val_loss, val_accuracy = model.evaluate(x=val_data,  y=val_label)
print(val_loss)
print(val_accuracy)

leng = val_label.shape[0]

i = 0
correct = 0
wrong = 0

preds = list()
real = list()

with tqdm(total=leng) as pbar:
    for sample, label in zip(val_data, val_label):
        item = np.expand_dims(sample, axis=0)
        #print(item)
        #print(item.shape)
        out = model.predict(item)
        y_pred = np.argmax(out)
        out[np.where(out!=np.max(out))] = 0
        out[np.where(out==np.max(out))] = 1
        y = np.argmax(label)
        if y_pred == y:
            correct += 1
        else:
            wrong += 1
        #print(y_pred)
        #print(y)
        real.append(y)
        preds.append(y_pred)
        pbar.update(1)

print(len(preds))
print(len(real))

y_test = np.empty([1, 1, 109], dtype=int)
y_test = np.vstack(real)

y_pred = np.empty([1, 1, 109], dtype=int)
y_pred = np.vstack(preds)

print(y_test.shape)
print(y_pred.shape)

#results = np.concatenate((y_test, y_pred), axis=1)
results = np.column_stack((y_test, y_pred))
print(results)
print(results.shape)

from statistics import mean

frrs = list(0 for x in range(0, 109))
fars = list(0 for x in range(0, 109))
#(test, pred)
for s in range(0, 109):
    #print("Class")
    #print(s)
    resultsTempTrue  = results[np.where(results[:,0]==s)]  #label 
    resultsTempFalse = results[np.where(results[:,1]==s)]  #predetti bene
    #print(resultsTempFalse)
    #print(resultsTempFalse.shape)
    cnt4 = resultsTempFalse.shape[0] #numero predetti bene
    errors = resultsTempFalse[resultsTempFalse[:,0]!=s]  #predetti male
    #print(errors.shape)
    #print(errors)
    cnt3 = errors.shape[0]  #numero predetti male
    cnt  = 0
    cnt1 = 0 
    #print("shape")
    #print(resultsTempTrue.shape[0])
    for i in range(0, resultsTempTrue.shape[0]):
        cnt1 += 1
        if resultsTempTrue[i][0] != resultsTempTrue[i][1]:
            cnt += 1
    #print("cnt1")
    #print(cnt1)
    frrTemp = cnt3/cnt1 
    frrs.append(frrTemp)

    farTemp = cnt3/cnt4
    fars.append(farTemp)
    #print(frrTemp)
    #print(farTemp)

FAR_mean = mean(fars)
FRR_mean = mean(frrs)
eer = (FRR_mean + FAR_mean)/2
print(FAR_mean)
print(FRR_mean)
print(eer)