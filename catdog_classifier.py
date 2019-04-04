import cv2
import numpy as np
from keras.models import load_model
import easygui

def process():
    path = easygui.fileopenbox()
    model = load_model('savedModel.h5')
    img = cv2.imread(path)
    img = cv2.resize(img,(64,64))
    img = np.reshape(img,[1,64,64,3])

    classes = model.predict_classes(img)
    #print(classes)
    if classes[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    easygui.msgbox(prediction)
    
if __name__ =='__main__':
        process()