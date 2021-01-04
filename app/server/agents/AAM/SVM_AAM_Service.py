from menpo.visualize import print_progress
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68_trimesh
import menpo.io as mio
import pandas as pd
from keras.models import Sequential, Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from joblib import dump, load

from app.server.agents.ImageProcessor import ImageProcessor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from app.server.agents.AAM.AAM_use import AAM_Class
import numpy as np


class SVM_AAM_Service:

    def __init__(self):
        self.path_to_training_images = 'C:/Users/Catalin/Desktop/facultate/licenta/DB/lfpw/trainset/'
        self.path_to_store_aam_model = '../cache_results/fitter_save'
        self.path_to_store_svm_model = '../cache_results/svm_model.joblib'
        self.path_to_my_picture = "D:\\ac\\FERforReviews\\app\\server\\agents\\test_images\\picture2.jpg"
        self.path_to_their_picture = "C:/Users/Catalin/Desktop/facultate/licenta/DB/lfpw/testset/image_0018.png"
        self.path_to_putin_picture = "C:\\Users\\Catalin\\Desktop\\facultate\\Semestru_5\\MIRPR\\laborator\\aplicatie-git\\New folder\\mirpr-calculafectiv\\app\\server\\agents\\test_images\\putin.png"
        self.path_to_bd = "D:\\ac\\FERforReviews\\app\\server\\agents\\db.csv"
        self.aam = AAM_Class(self.path_to_training_images, self.path_to_store_aam_model)
        self.aam_model = ""
        self.svm_model = ""

    def processData(self, path):
        # read data
        df = pd.read_csv(path, sep=' ')
        # df = df[:10]

        # shuffle data
        df = df.sample(frac=1)
        # split data into training,validation and test data(70,10,20)
        df.isnull().sum()
        print(df.describe)
        df['pixels'] = df['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
        x_train = np.vstack(df['pixels'][0:int(len(df) * 0.7)].values)
        y_train = np.array(df["emotion"][0:int(len(df) * 0.7)])

        x_valid = np.vstack(df["pixels"][int(len(df) * 0.7):int(len(df) * 0.8)].values)
        y_valid = np.array(df["emotion"][int(len(df) * 0.7):int(len(df) * 0.8)])

        x_test = np.vstack(df["pixels"][int(len(df) * 0.8):len(df)].values)
        y_test = np.array(df["emotion"][int(len(df) * 0.8):len(df)])

        # normalize x
        # x_train=np.array(x_train)/255.0
        # x_valid=np.array(x_valid)/255.0
        # x_test=np.array(x_test)/255.0

        # reshape x into 2D
        N = len(x_train)
        x_train = x_train.reshape(N, 256, 256, 1)
        N = len(x_valid)
        x_valid = x_valid.reshape(N, 256, 256, 1)
        N = len(x_test)
        x_test = x_test.reshape(N, 256, 256, 1)

        # reshape y into 2D
        num_class = len(set(y_test))
        Y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
        Y_valid = (np.arange(num_class) == y_valid[:, None]).astype(np.float32)
        Y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)

        return x_train, y_train, x_valid, y_valid, x_test, y_test, Y_train, Y_valid, Y_test

    def processData2(self, path):
        # read data
        df = pd.read_csv(path, sep=' ')
        # df = df[:40]

        # shuffle data
        df = df.sample(frac=1)
        # split data into training,validation and test data(70,10,20)
        df.isnull().sum()
        print(df.describe)
        df['pixels'] = df['pixels'].apply(lambda im: np.fromstring(im, sep=' '))
        x_train = np.vstack(df['pixels'][0:int(len(df) * 0.7)].values)
        y_train = np.array(df["emotion"][0:int(len(df) * 0.7)])

        x_valid = np.vstack(df["pixels"][int(len(df) * 0.7):int(len(df) * 0.8)].values)
        y_valid = np.array(df["emotion"][int(len(df) * 0.7):int(len(df) * 0.8)])

        x_test = np.vstack(df["pixels"][int(len(df) * 0.8):len(df)].values)
        y_test = np.array(df["emotion"][int(len(df) * 0.8):len(df)])

        # normalize x
        # x_train=np.array(x_train)/255.0
        # x_valid=np.array(x_valid)/255.0
        # x_test=np.array(x_test)/255.0

        # reshape x into 2D
        N = len(x_train)
        x_train = x_train.reshape(N, 256, 256, 1)
        N = len(x_valid)
        x_valid = x_valid.reshape(N, 256, 256, 1)
        N = len(x_test)
        x_test = x_test.reshape(N, 256, 256, 1)

        # reshape y into 2D
        num_class = len(set(y_test))
        Y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
        Y_valid = (np.arange(num_class) == y_valid[:, None]).astype(np.float32)
        Y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)

        return x_train, y_train, x_valid, y_valid, x_test, y_test, Y_train, Y_valid, Y_test

    # AAM-AAM-AAM-AAM-AAM-AAM-AAM-AAM-AAM

    def loadAAM(self):
        self.aam_model = self.aam.loadModel(self.path_to_store_aam_model)

    def saveAAM(self):
        self.aam.saveModel(self.path_to_store_aam_model, self.aam_model)

    def trainAAM(self):
        self.aam_model = self.aam.train()

    def interpretEmpotion(self, id):
        if id == 0:
            return "angry"
        if id == 1:
            return "disgust"
        if id == 2:
            return "fearful"
        if id == 3:
            return "happy"
        if id == 4:
            return "neutral"
        if id == 5:
            return "sad"
        if id == 6:
            return "surprise"
        if id == 7:
            return "????"

    def trainAAMWithImages(self, training_images):
        self.aam_model = self.aam.trainWithImages(training_images)

    def useAAM(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test, Y_train, Y_valid, Y_test = self.processData(self.path_to_bd)
        self.loadAAM()

        colorMioImage = mio.import_image(self.path_to_putin_picture)
        colorResult = self.aam.getResulsFromColorImage(self.aam_model, colorMioImage)
        self.aam.print_things(self.aam_model, colorResult)

        grayResult = self.aam.getResulsFromGrayNdArray(self.aam_model, x_train[0])
        a = grayResult.final_shape.points
        self.aam.print_things(self.aam_model, grayResult)

    def prepare_features(self, x_train, x_valid, x_test, y_train, y_valid, y_test):

        feat_x_train = []
        feat_y_train = []
        indice = 0
        for x, y in zip(x_train, y_train):
            print("x_train")
            print("indice: " + str(indice))
            print("emotion: " + self.interpretEmpotion(y_train[indice]))
            point = self.aam.getLandmarksFromGrayNdArray(self.aam_model, x)
            list = []
            if (len(point) != 0):
                for p in point:
                    list.append(p[0])
                    list.append(p[1])
                feat_x_train.append(list)
                feat_y_train.append(y)
            indice = indice + 1

        feat_x_valid = []
        feat_y_valid = []
        indice = 0
        for x, y in zip(x_valid, y_valid):
            print("x_valid")
            print("indice: " + str(indice))
            print("emotion: " + self.interpretEmpotion(y_valid[indice]))
            indice = indice + 1
            point = self.aam.getLandmarksFromGrayNdArray(self.aam_model, x)
            list = []
            if (len(point) != 0):
                for p in point:
                    list.append(p[0])
                    list.append(p[1])
                feat_x_valid.append(list)
                feat_y_valid.append(y)

        feat_x_test = []
        feat_y_test = []
        indice = 0
        for x, y in zip(x_test, y_test):
            print("x_test")
            print("indice: " + str(indice))
            print("emotion: " + self.interpretEmpotion(y_test[indice]))
            indice = indice + 1
            point = self.aam.getLandmarksFromGrayNdArray(self.aam_model, x)
            list = []
            if (len(point) != 0):
                for p in point:
                    list.append(p[0])
                    list.append(p[1])
                feat_x_test.append(list)
                feat_y_test.append(y)

        return feat_x_train, feat_y_train, feat_x_valid, feat_y_valid, feat_x_test, feat_y_test

    # AAM-AAM-AAM-AAM-AAM-AAM-AAM-AAM-AAM

    # SVM-SVM-SVM-SVM-SVM-SVM-SVM-SVM-SVM

    def loadSVM(self):
        self.svm_model = load(self.path_to_store_svm_model)

    def saveSVM(self):
        d = 2

    def evaluate(self, model, X, Y):
        predicted_Y = model.predict(X)
        accuracy = accuracy_score(Y, predicted_Y)
        return accuracy

    def matrix(self, svm_model, feat_valid, y_valid, feat_test, y_test):

        validation_accuracy = self.evaluate(svm_model, feat_valid, y_valid)
        print("  - validation accuracy = {0:.1f}%".format(validation_accuracy * 100))
        valid_labels = svm_model.predict(feat_valid)
        print(classification_report(y_valid, valid_labels))
        mat = confusion_matrix(y_valid, valid_labels)
        print(mat)

        test_accuracy = self.evaluate(svm_model, feat_test, y_test)
        print("  - test accuracy = {0:.1f}%".format(test_accuracy * 100))
        test_labels = svm_model.predict(feat_test)
        print(classification_report(y_test, test_labels))
        mat = confusion_matrix(y_test, test_labels)
        print(mat)

    def fitEvaluateSVC(self, path_to_store_svm_model, feat_train, feat_valid, feat_test, y_train, y_valid, y_test):

        svm_model_1 = LinearSVC(C=100.0, max_iter=100000)
        svm_model_1.fit(feat_train, y_train)
        print('fitting done !!!')
        dump(svm_model_1, path_to_store_svm_model)

        self.matrix(svm_model_1, feat_valid, y_valid, feat_test, y_test)

        return svm_model_1

    def trainSVM(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test, Y_train, Y_valid, Y_test = self.processData2(
            self.path_to_bd)
        feat_x_train, feat_y_train, feat_x_valid, feat_y_valid, feat_x_test, feat_y_test = self.prepare_features(
            x_train, x_valid, x_test, y_train, y_valid, y_test)

        self.svmModel = self.fitEvaluateSVC(self.path_to_store_svm_model, feat_x_train, feat_x_valid, feat_x_test,
                                            feat_y_train, feat_y_valid, feat_y_test)

    def Predict(self, featureModel, classifyModel, x):
        featureModel = Model(inputs=featureModel.input, outputs=featureModel.get_layer('dense_one').output)
        x = featureModel.predict(x)
        predictedY = classifyModel.predict(x)
        print(predictedY)

    def useSVM(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test, Y_train, Y_valid, Y_test = self.processData2(
            self.path_to_bd)
        self.loadAAM()
        self.loadSVM()

        colorMioImage = mio.import_image(self.path_to_my_picture)

        colorResult = self.aam.getResulsFromColorImage(self.aam_model, colorMioImage)
        a = colorResult.final_shape.points
        self.aam.print_things(self.aam_model, colorResult)

        list = []
        xa = []
        for p in a:
            list.append(p[0])
            list.append(p[1])
        xa.append(list)

        predictedYa = self.svm_model.predict(xa)
        print(self.interpretEmpotion(predictedYa))

        grayResult = self.aam.getResulsFromGrayNdArray(self.aam_model, x_train[0])
        b = grayResult.final_shape.points
        self.aam.print_things(self.aam_model, grayResult)

        list = []
        xb = []
        for p in b:
            list.append(p[0])
            list.append(p[1])

        xb.append(list)

        predictedYb = self.svm_model.predict(xb)
        print(self.interpretEmpotion(predictedYb))

    def predictEmotionForFrame(self, imagePath):
        self.loadAAM()
        self.loadSVM()

        colorMioImage = mio.import_image(imagePath)

        colorResult = self.aam.getResulsFromColorImage(self.aam_model, colorMioImage)
        a = colorResult.final_shape.points
        self.aam.print_things(self.aam_model, colorResult)

        list = []
        xa = []
        for p in a:
            list.append(p[0])
            list.append(p[1])
        xa.append(list)

        predictedYa = self.svm_model.predict(xa)
        emotion = self.interpretEmpotion(predictedYa)
        print(emotion)
        return emotion

    # SVM-SVM-SVM-SVM-SVM-SVM-SVM-SVM-SVM

    # PRINT-IMAGE

    def printUrlPNGJPGColorImage(self, url):
        plt.imshow(mpimg.imread(url))
        plt.show()

    def printGrayImage(self, grayImage):
        processor = ImageProcessor()
        listPixels = processor.processGrayImageForDisplay(grayImage)

        plt.imshow(listPixels, cmap="gray")
        plt.show()

    # PRINT-IMAGE

    # RUN-RUN-RUN-RUN-RUN-RUN-RUN-RUN-RUN

    def run(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test, Y_train, Y_valid, Y_test = self.processData(self.path_to_bd)
        self.loadAAM()
        self.prepare_features(x_train, x_valid, x_test, y_train, y_valid, y_test)
        # self.trainSVM(x_train, x_valid, x_test, y_train, y_valid, y_test)

    def runTrain(self):
        training_images = self.loadImagesForAAM()
        self.trainAAMWithImages(training_images)
        self.saveAAM()

        self.loadAAM()

        self.trainSVM()
        self.saveSVM()

    def runLoad(self):
        self.loadAAM()
        self.loadSVM()

    def runPrintImages(self):
        # self.printUrlPNGJPGColorImage(self.path_to_my_picture)
        # self.printUrlPNGJPGColorImage(self.path_to_putin_picture)
        self.printUrlPNGJPGColorImage(self.path_to_their_picture)

        # self.printUrlPNGJPGGrayScaleImage(self.path_to_my_picture)
        # self.printUrlPNGJPGGrayScaleImage(self.path_to_putin_picture)
        # self.printUrlPNGJPGGrayScaleImage(self.path_to_their_picture)
        x_train, y_train, x_valid, y_valid, x_test, y_test, Y_train, Y_valid, Y_test = self.processData(self.path_to_bd)

        self.printGrayImage(x_train[0])

    # RUN-RUN-RUN-RUN-RUN-RUN-RUN-RUN-RUN

    # load-images

    def loadImages(self, path_to_training_images):
        training_images = []
        for img in print_progress(mio.import_images(path_to_training_images, verbose=True)):
            # convert to greyscale
            if img.n_channels == 3:
                img = img.as_greyscale()
            # crop to landmarks bounding box with an extra 20% padding
            img = img.crop_to_landmarks_proportion(0.2)
            # rescale image if its diagonal is bigger than 400 pixels
            d = img.diagonal()
            if d > 400:
                img = img.rescale(400.0 / d)
            # define a TriMesh which will be useful for Piecewise Affine Warp of HolisticAAM
            labeller(img, 'PTS', face_ibug_68_to_face_ibug_68_trimesh)
            # append to list
            training_images.append(img)
        return training_images

    def loadImagesForAAM(self):
        self.path_to_training_images1 = 'C:/Users/Catalin/Desktop/facultate/licenta/DB/AAM/lfpw/trainset'
        self.path_to_training_images2 = 'C:/Users/Catalin/Desktop/facultate/licenta/DB/AAM/lfpw/testset'
        self.path_to_training_images3 = 'C:/Users/Catalin/Desktop/facultate/licenta/DB/AAM/ibug'
        self.path_to_training_images4 = 'C:/Users/Catalin/Desktop/facultate/licenta/DB/AAM/helen/trainset'
        self.path_to_training_images5 = 'C:/Users/Catalin/Desktop/facultate/licenta/DB/AAM/helen/testset'
        self.path_to_training_images6 = 'C:/Users/Catalin/Desktop/facultate/licenta/DB/AAM/afw'

        images = []

        images1 = self.loadImages(self.path_to_training_images1)
        images2 = self.loadImages(self.path_to_training_images2)
        images3 = self.loadImages(self.path_to_training_images3)

        images4 = self.loadImages(self.path_to_training_images4)
        images5 = self.loadImages(self.path_to_training_images5)

        images6 = self.loadImages(self.path_to_training_images6)

        for img in images1:
            images.append(img)
        for img in images2:
            images.append(img)

        for img in images3:
            images.append(img)

        for img in images4:
            images.append(img)
        for img in images5:
            images.append(img)

        for img in images6:
            images.append(img)

        return images

    # load-images


# service = SVM_AAM_Service()
# service.run()
# service.runPrintImages()
# service.useAAM()
# service.loadImagesForAAM()
# service.runTrain()
# service.useSVM()
# service.predictEmotionForFrame("D:\\ac\\FERforReviews\\app\\server\\agents\\test_images\\putin.png")
