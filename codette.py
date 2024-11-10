import os
import urllib.request
import cv2
import dlib

dat_url = 'http://0x0.c0de.wtf/sillyprojects/shape_predictor_68_face_landmarks.dat'
png_url = 'http://0x0.c0de.wtf/sillyprojects/a.png'

dat_path = 'shape_predictor_68_face_landmarks.dat'
png_path = 'a.png'

def download_file(url, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading required file: {file_path}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Download: {file_path} done.")

download_file(dat_url, dat_path)
download_file(png_url, png_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(dat_path)

def add_bow(image, position):
    bow = cv2.imread(png_path, -1) 
    bow = cv2.resize(bow, (50, 50)) 

    x, y = position
    x -= -10
    y -= 10 
    
    for i in range(bow.shape[0]):
        for j in range(bow.shape[1]):
            if bow[i, j][3] != 0:
                alpha = bow[i, j][3] / 255.0
                image[y + i, x + j] = alpha * bow[i, j][:3] + (1 - alpha) * image[y + i, x + j]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        bow_position = (landmarks.part(27).x, landmarks.part(27).y - 80)

        add_bow(frame, bow_position)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27: #ESC
        break

cap.release()
cv2.destroyAllWindows()
