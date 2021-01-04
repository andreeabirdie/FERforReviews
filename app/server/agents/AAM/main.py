from app.server.agents.AAM.SVM_AAM_Service import SVM_AAM_Service
import cv2
import os

FPS = 15
cap = cv2.VideoCapture('D:\\ac\\FERforReviews\\app\\server\\agents\\video_sample\\02-02-04-02-02-01-01.mp4')
cap.set(cv2.CAP_PROP_FPS, FPS)

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print('Error: Creating directory of data')

currentFrame = 0
while True:
    # Capture frame-by-frame
    success, frame = cap.read()

    # Saves image of the current frame in jpg file
    name = './data/frame' + str(currentFrame) + '.jpg'
    print('Creating...' + name)
    if success:
        cv2.imwrite(name, frame)
    else:
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

service = SVM_AAM_Service()
emotions = []
for i in range(currentFrame):
    if i % 10 == 0:
        emotion = service.predictEmotionForFrame('./data/frame' + str(i) + '.jpg')
        emotions.append(emotion)

print(emotions)
