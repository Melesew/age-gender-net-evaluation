# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import subprocess

TRAINSET = "haarcascade_frontalface_alt.xml"
DOWNSCALE = 4
gender = ""
web_cam = cv2.VideoCapture(-1)
classifier = cv2.CascadeClassifier(TRAINSET)

if web_cam.isOpened():
    rval, frame = web_cam.read()
else:
    rval = False
    print ("here")

while rval:
    # detect faces and draw bounding boxes
    minisize = (frame.shape[1] // DOWNSCALE, frame.shape[0] // DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = classifier.detectMultiScale(miniframe)

    gender = ""

    if len(faces) == 0:
        cv2.putText(frame, "face not detected.", (250, 250),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255))

    for f in faces:
        x, y, w, h = [v * DOWNSCALE for v in f]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))
        cv2.putText(frame, gender, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255))

    cv2.putText(frame, "Press C, to know your gender\n"+ gender , (5, 25),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 255))

    cv2.putText(frame, gender, (15, 35),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (155, 255, 255))

    cv2.imshow("preview", frame)

    # get next frame
    rval, frame = web_cam.read()
    if cv2.waitKey(1) & 0xFF == ord('c') and len(faces) != 0:
        cv2.imwrite("cropped/1.jpg", frame)
        subprocess.check_output(['python3', 'facecrop_demo.py', '--image', 'cropped/1.jpg'])

        result = subprocess.check_output(['python3', 'predict.py', '--model', 'models/gen.h5', '--image', 'cropped/1_cropped_.jpg'])

        text = None
        if ord(result[:-1]) == ord('0'):
            text = "Female"
        elif ord(result[:-1]) == ord('1'):
            text = "male"

        gender = text
        if gender != None:
            cv2.putText(frame, "You are "+gender+ " aren't you? Press any key to check again.", (15, 35),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (155, 255, 255))

            cv2.imshow("preview", frame)

            cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
