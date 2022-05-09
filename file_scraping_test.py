from email import message
import os
import methods
from cv2 import convertScaleAbs
import glob
import numpy
import pandas as pd
import matplotlib.pyplot as plt

# MAX_LEN = 20
# s = "D:\\Datasets\\facebook\\facebook-json-all-low"
# dataset = methods.make_datasetv2(s)

# methods.write_dataset(dataset)
df = pd.read_csv('Final_version\model_history_log_final.csv')

loss_train = df['loss']
loss_val = df['val_loss']
epochs = df['epoch']

plt.plot(epochs, loss_train, 'g', label='Training loss',color='blue')
plt.plot(epochs, loss_val, 'b', label='Validation loss',color='orange')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('Final_version\\graphs\\training_val_loss.png')

plt.show()

plt.savefig('foo.png')
acc_val = df['val_accuracy']
acc_train = df['accuracy']
epochs = df['epoch']

plt.plot(epochs, acc_val, 'g', label='Validation accuracy',color='orange')
plt.plot(epochs, acc_train, 'b', label='Training accuracy',color='blue')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('Final_version\\graphs\\training_val_acc.png')

plt.show()
plt.plot(epochs, acc_train, 'g', label='Validation accuracy',color='orange')
plt.plot(epochs, loss_train, 'b', label='Training accuracy',color='blue')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('Final_version\\graphs\\training_acc_training_loss.png')
