import cv2
import numpy as np
#import tensorflow as tf
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
import time
from tensorflow.keras.models import load_model
import textwrap

# CLASS_NAMES = [
# 'Pepper__bell___Bacterial_spot',
# 'Pepper__bell___healthy',
# 'Potato___Early_blight',
# 'Potato___Late_blight',
# 'Potato___healthy',
# 'Tomato_Bacterial_spot',
# 'Tomato_Early_blight',
# 'Tomato_Late_blight',
# 'Tomato_Leaf_Mold',
# 'Tomato_Septoria_leaf_spot',
# 'Tomato_Spider_mites_Two_spotted_spider_mite',
# 'Tomato__Target_Spot',
# 'Tomato__Tomato_YellowLeaf__Curl_Virus',
# 'Tomato__Tomato_mosaic_virus',
# 'Tomato_healthy'
# ]

CLASS_NAMES = ['BellPepper Bacterial_spot','Potato Early blight','BellPepper healthy','Potato Late blight','Potato healthy','Tomato Bacterialspot','Tomato Early blight','Tomato Late blight','Tomato LeafMold','Tomato Septoria Leafspot','Tomato Spidermites Twospotted','Tomato TargetSpot','Tomato YellowLeaf Curl Virus','Tomato Mosaic Virus','Tomato Healthy']

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)



def Char_recognition():
    Mnist = load_model('Dense_Net_2.h5')
    input_type = input('To load a video file press 0 \nTo use live camera press 1\n')
    out = cv2.VideoWriter('Plant_Disease_Saved_Video_Output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (256,256))
    out_squared = 'Squared_Image'
    if(input_type == '0'):
        path = input('Please enter the path to the video\n')
        video = cv2.VideoCapture(path)
    else:
        video = cv2.VideoCapture(0)
        out = cv2.VideoWriter('Plant_Disease_Live_Video_Output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (256,256))
    start_time = time.time()
    while (video.isOpened() and int(time.time() - start_time) < 50 ):
        ret, frame = video.read()

        if ret == True:
            Image = cv2.resize(frame, (256,256), interpolation=cv2.INTER_AREA)
            Gray_Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
            Gray_Image_blur = cv2.GaussianBlur(Gray_Image, (5,5), cv2.BORDER_DEFAULT)

            Final_Gray_Image = cv2.adaptiveThreshold(Gray_Image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
            Canny_Edge = cv2.Canny(Final_Gray_Image, 100, 200)

            #cv2.imshow('Canny Edge', Final_Gray_Image)

            contours, hierarchy = cv2.findContours(Canny_Edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for i, val in enumerate(contours):
                
                [x,y,w,h] = cv2.boundingRect(val)
                # border_image = cv2.rectangle(Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # #border_image = cv2.putText(img = border_image, text = str(prediction), org = (x, y-5), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 3, color=(255, 0, 0), thickness = 1, lineType = cv2.LINE_AA)
                # cv2.imshow('Border_image', border_image)

                if h > 70 and w > 70:
                    cropped_image = Image[y:y+h, x:x+w]
                    cropped_image = cv2.resize(cropped_image, (64,64), interpolation=cv2.INTER_LINEAR)
                    x_test = np.zeros((1, 64, 64, 3))
                    x_test[0] = resize_image(cropped_image, (64, 64))
                    x_test[0] = x_test[0]/255
                    #cv2.imshow('Cropped_Image', cropped_image)
                    # cropped_image = cropped_image / 255.0
                    #prediction = CLASS_NAMES[Mnist.predict(cropped_image.reshape(1,64,64,3)).argmax()]

                    prediction = Mnist.predict(x_test)
                    prediction = CLASS_NAMES[np.argmax(prediction[0])]


                    print(prediction)
                    border_image = cv2.rectangle(Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    border_image = cv2.putText(img = border_image, text = str(prediction), org = (x, y-5), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, color=(0, 0, 0), thickness = 1, lineType = cv2.LINE_AA)
                    cv2.imshow('Border_image', border_image)
                    out.write(border_image)

        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    video_cap.release()
    cv2.destroyAllWindows()

def predict_id():
    Char_recognition()


# train_model()
predict_id()