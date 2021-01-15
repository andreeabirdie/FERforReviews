from app.server.agents.AAM.SVM_AAM_Service import SVM_AAM_Service
import cv2
import os
import matplotlib.pyplot as plotter


def reviewResults(listOfEmotions):
    angry = 0
    disgust = 0
    fearful = 0
    happy = 0
    neutral = 0
    sad = 0
    surprise = 0
    positiveEmotion = 0
    negativeEmotion = 0
    for emotion in listOfEmotions:
        if emotion == 'angry':
            angry = angry + 1
            negativeEmotion = negativeEmotion + 1
        if emotion == 'disgust':
            disgust = disgust + 1
            negativeEmotion = negativeEmotion + 1
        if emotion == 'fearful':
            fearful = fearful + 1
            negativeEmotion = negativeEmotion + 1
        if emotion == 'happy':
            happy = happy + 1
            positiveEmotion = positiveEmotion + 1
        if emotion == 'neutral':
            neutral = neutral + 1
            positiveEmotion = positiveEmotion + 1
        if emotion == 'sad':
            sad = sad + 1
            negativeEmotion = negativeEmotion + 1
        if emotion == 'surprise':
            surprise = surprise + 1
            positiveEmotion = positiveEmotion + 1

    angry = angry * 100 / len(listOfEmotions)
    disgust = disgust * 100 / len(listOfEmotions)
    fearful = fearful * 100 / len(listOfEmotions)
    sad = sad * 100 / len(listOfEmotions)
    happy = happy * 100 / len(listOfEmotions)
    neutral = neutral * 100 / len(listOfEmotions)
    surprise = surprise * 100 / len(listOfEmotions)

    pieLabels = ['angry', 'disgust', 'fearful', 'sad', 'happy', 'neutral', 'surprise']
    populationShare = [angry, disgust, fearful, sad, happy, neutral, surprise]

    i = 0
    while i < len(populationShare):
        if populationShare[i] == 0:
            populationShare.remove(populationShare[i])
            pieLabels.remove(pieLabels[i])
        else:
            i = i + 1

    figureObject, axesObject = plotter.subplots()

    axesObject.pie(populationShare, labels=pieLabels, autopct='%1.2f', startangle=90)

    axesObject.axis('equal')

    plotter.show()

    print('')
    print('Angry:               ' + str(angry) + '%')
    print('Disgust:             ' + str(disgust) + '%')
    print('Fearful:             ' + str(fearful) + '%')
    print('Sad:                 ' + str(sad) + '%')
    print('Happy:               ' + str(happy) + '%')
    print('Neutral:             ' + str(neutral) + '%')
    print('Surprise:            ' + str(surprise) + '%')

    print('')
    print('')
    print('Negative Emotion:    ' + str(negativeEmotion * 100 / len(listOfEmotions)) + '%')
    print('Positive Emotion:    ' + str(positiveEmotion * 100 / len(listOfEmotions)) + '%')


def divideVideoIntoFrames(pathToVideo):
    FPS = 15
    cap = cv2.VideoCapture(pathToVideo)
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
    return currentFrame


service = SVM_AAM_Service()
emotions = []
# currentFrame = divideVideoIntoFrames('D:\\ac\\FERforReviews\\app\\server\\agents\\video_sample\\01-02-03-01-01-01-01.mp4')
# currentFrame = 137
# for i in range(currentFrame):
#     if currentFrame % 10 == 0:
#         emotion = service.predictEmotionForFrame('D:\\ac\\FERforReviews\\app\\server\\agents\\AAM\\data\\frame' + str(i) + '.jpg')
#         emotions.append(emotion)
# if len(emotions) > 0:
#     reviewResults(emotions)
print(service.predictEmotionForFrame('./data/frame120.jpg'))
