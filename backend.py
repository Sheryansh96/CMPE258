from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from PIL import Image
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2

app = FastAPI()
mod = tf.keras.models.load_model("Dense_Net_2.h5")
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CLASS_NAMES = [
# "Pepper__bell___Bacterial_spot",
# "Pepper__bell___healthy",
# "Potato___Early_blight",
# "Potato___Late_blight",
# "Potato___healthy",
# "Tomato_Bacterial_spot",
# "Tomato_Early_blight",
# "Tomato_Late_blight",
# "Tomato_Leaf_Mold",
# "Tomato_Septoria_leaf_spot",
# "Tomato_Spider_mites_Two_spotted_spider_mite",
# "Tomato__Target_Spot",
# "Tomato__Tomato_YellowLeaf__Curl_Virus",
# "Tomato__Tomato_mosaic_virus"
# "Tomato_healthy",
# ]

#CLASS_NAMES = ['Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']
CLASS_NAMES = ['Pepper__bell___Bacterial_spot','Potato___Early_blight','Pepper__bell___healthy','Potato___Late_blight','Potato___healthy','Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight','Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot','Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus','Tomato_healthy']


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

@app.get("/data")
async def ping():
    return "Server Running"

@app.post("/predict")
async def pic_predict(
    file: UploadFile = File(...)
):
    img = read_file_as_image(await file.read())
    x_test = np.zeros((1, 64, 64, 3))
    x_test[0] = resize_image(img, (64, 64))
    x_test[0] = x_test[0]/255
    
    #print(x_test)
    img_batch = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
    predictions = mod.predict(x_test)
    

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)