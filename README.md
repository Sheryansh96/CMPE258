# CMPE258 - Plant disease classification using Dense Net

## Project team:
- Shreyansh Upadhyay - 015226614 
- Anurag Gajam - 016003377
- Shanmukh Boddu - 016005743 
- Roopa Chiluvuri – 016005795


### Steps to check running of the program 
1. Download the zip file

2. Setup for Python:
 Ensure the following versions match
 - Python : 3.7.13
 - Opencv-python : 4.1.1.26 
 - Tensorflow : 2.9.1
 - Node : 14.7.3


3. Training the model:
 - To train the model run the Preprocessing_Model_training.ipynb file. 2. The output of each cell is already present in the notebook.

4. Setup for running the backend:
 Install the following packages using pip:-
  - fastapi
  - uvicorn
  - python-multipart
  - pillow
  - tensorflow-serving-api==2.5.0 
  - matplotlib
  - numpy

#### Install the npm packages using 
 - cd frontend
 - npm install --from-lock-json 
 - npm audit fix


5. Run backend.py using python backend.py

6. Then run the frontend application using npm start which will be running on localhost - 3000.

#### Output

<img width="799" alt="image" src="https://user-images.githubusercontent.com/100038612/204424262-40c1512d-9bd6-4f24-978f-43bc915518ad.png">

Select the image to test and the output will be displayed as below

<img width="800" alt="image" src="https://user-images.githubusercontent.com/100038612/204424664-5288c01b-17bb-42f3-86d7-ccaedd216e4a.png">


7. To run the live video plant disease detection use:-
 - Run python live_video.py
 - Being the image of the plant leaf closer to the camera till the time it is detected and the output of the plant will be shown in the console.
 
