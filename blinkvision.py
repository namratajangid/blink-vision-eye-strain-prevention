# Imports
import cv2
import dlib
import time
from scipy.spatial import distance
from imutils import face_utils
from win10toast import ToastNotifier

# To show notifications
toastNotifier = ToastNotifier()

# To capture live video
cap = cv2.VideoCapture(0)

# Facial landmarks detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/My Data/Documents/shape_predictor_68_face_landmarks.dat')

# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye = (A + B) / (2.0 * C)
    return eye

count = 0
total = 0

# Custom duration to be set by the user
print("We suggest setting the duration as 20 minutes. Looking at an object placed 20 feet away for 20 seconds every 20 minutes promotes good eye health.")
durationMinutes = int(input("Set your Blink Vision notification duration in minutes: "))
print("Blink Vision will notify you every "+str(durationMinutes)+" minutes if you are not blinking enough. The ideal blink rate is at least 15 blinks per minute.")

start_time = time.time()
duration = 60*durationMinutes
idealBlinkRate = 15

while True:

    while True:

        current_time = time.time()
        elapsed_time = current_time - start_time

        success, img = cap.read()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(imgGray)

        for face in faces:
            landmarks = predictor(imgGray, face)

            landmarks = face_utils.shape_to_np(landmarks)
            leftEye = landmarks[42:48]
            rightEye = landmarks[36:42]

            leftEye = eye_aspect_ratio(leftEye)
            rightEye = eye_aspect_ratio(rightEye)

            eye = (leftEye + rightEye) / 2.0

            if eye < 0.3:
                count += 1
            else:
                if count >= 3:
                    total += 1

                count = 0

        cv2.putText(img, "Blink Count: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        if elapsed_time > duration:

            avgBlinkRate = total/durationMinutes

            start_time = time.time()
            if (avgBlinkRate) < idealBlinkRate:
                toastNotifier.show_toast("BLINK VISION", " Please rest your eyes for a bit. Your blink rate is "+ str(avgBlinkRate) +" blinks per minutes, which is less than the ideal blink rate of " + str(idealBlinkRate) +" blinks per minute!", duration=10)
            total = 0
            count = 0

