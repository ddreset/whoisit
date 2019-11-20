from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.models import *
from keras.layers import Dense, Activation, Conv2D, concatenate
from keras.layers import Flatten, Dropout, ZeroPadding2D, BatchNormalization, UpSampling2D, MaxPooling2D
from keras.models import save_model, load_model
from keras.models import Model
from keras.optimizers import *

def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

model = unet()
model = load_model('crop_mosaic_model.h5')

# input size of model is (256,256)
def read_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def preprocess(imgArray):
    imgArray = cv2.blur(imgArray, (2,2))
    edges = cv2.Canny(imgArray, 0, 250)
    edges[edges > 0] = 1
    edges = np.asarray(edges, dtype=int)
    model_input = []
    model_input.append(np.expand_dims(edges, axis=2))
    model_input = np.asarray(model_input)
    return model_input

def get_candidates(boxArray):
    Rows = [0]
    Columns = [0]
    max_row = boxArray.shape[0]-1
    max_col = boxArray.shape[1]-1
    rowThreshold = (max_col+1)*0.25
    colThreshold = (max_row+1)*0.25

    # candidate rows
    for index in range(boxArray.shape[0]-2):
        # skip 0 and last row
        index = index + 1
        if index - Rows[-1] > 10 and np.sum(boxArray[index,:]) >= rowThreshold:
            Rows.append(index)

    if max_row - Rows[-1] > 20:
        Rows.append(max_row)

    # candidate columns
    for index in range(boxArray.shape[1]-2):
        # skip 0 and last column
        index = index + 1
        if index - Columns[-1] > 10 and np.sum(boxArray[:,index]) >= colThreshold:
            Columns.append(index)

    if max_col - Columns[-1] > 20:
        Columns.append(max_col)
    
    return Rows, Columns

def complete_box(boxArray):
    boxArray[0,:] = 1
    boxArray[255,:] = 1
    boxArray[:,0] = 1
    boxArray[:,255] = 1
    return boxArray

def get_bounding_box(boxArray,Rows,Columns):
    bounding_box = []
    for r1 in Rows[0:-1]:
        for c1 in Columns[0:-1]:
            found = False
            for r2 in Rows[1:]:
                if found:
                    break
                if r2 <= r1 or r2-r1<=20:
                    continue
                if np.sum(boxArray[r1:r2,c1])<=20:
                    break
                for c2 in Columns[1:]:
                    if c2 <= c1 or c2-c1<=20:
                        continue
                    if np.sum(boxArray[r1,c1:c2])<=20:
                        found = True
                        break
                    if np.sum(boxArray[r1:r2,c2])>20 and np.sum(boxArray[r2,c1:c2])>20:  
                        bounding_box.append([r1,c1,r2,c2])
                        found = True
                        break
                    if np.sum(boxArray[r1:r2,c2]) > 20 and np.sum(boxArray[r2,c1:c2])<=20:
                        break
    return bounding_box

def crop(image):
    imgArray = np.asarray(image.resize((256,256)))
    model_input = preprocess(imgArray)
    model_output = model.predict(model_input[0:])
    boxArray = np.squeeze(np.asarray(model_output))
    rows, columns = get_candidates(boxArray)
    boxArray = complete_box(boxArray)
    # [top,left,bottom,right]
    boxes = get_bounding_box(boxArray, rows, columns)
    # column scale, row scale
    scale = [image.size[0]/256,image.size[1]/256]
    cropped_img = []
    for box in boxes:
        # (left, top, right, bottom)
        coordinate = (box[1]*scale[0],box[0]*scale[1],box[3]*scale[0],box[2]*scale[1])
        cropped = image.crop(coordinate)
        cropped_img.append(cropped)
    return cropped_img

if __name__ == '__main__':
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Family_Cervidae_five_species.jpg/300px-Family_Cervidae_five_species.jpg"
    image = read_image(url)
    crop(image)