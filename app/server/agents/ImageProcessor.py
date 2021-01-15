import cv2
import numpy as np
# from skimage.measure._structural_similarity import compare_ssim as ssim
# from face_compare import images
# import face_recognition


class ImageProcessor(object):

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


    def generate(self, image_path, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            # print("Can't open image file")
            return 0

        faces = self.face_cascade.detectMultiScale(img, 1.1, 3)
        if (faces is None):
            print('Failed to detect face')
            return 0

        if (show_result):
            i=0
            pixels=np.zeros(1)
            for (x, y, w, h) in faces:
                #cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                crop_img = img[y:y+h, x:x+w]
                #cv2.imshow('img'+str(i), crop_img)
                #cv2.imwrite("tempface%d.png" % i,crop_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                pixels=self.processImage(crop_img)
        return pixels

    def processImage(self,img):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dim=(256,256)
        # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img',600,600)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img=cv2.resize(img,dim)
        pixels=np.asarray(img,dtype='float32')
        pixels=pixels/255.0
        return pixels.reshape(1,256,256,1)
        #return pixels

    def processGrayImageForDisplay(self,img):
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dim=(256,256)
        # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('img',600,600)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img=cv2.resize(img,dim)
        pixels=np.asarray(img,dtype='float32')
        #pixels=pixels/255.0
        return pixels.reshape(256,256)
        #return pixels

    def mse(self,imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        return err

'''
    def compare(self, target, image):
        """
        :param target: must be an ndarray
        :param image: must be an ndarray
        :return: true if they have the same face
        """

        target_face_encoding = face_recognition.face_encodings(target)[0]
        image_face_encoding = face_recognition.face_encodings(image)[0]

        # these line only shows the faces
        # target_face_locations = face_recognition.face_locations(target)
        # image_face_locations = face_recognition.face_locations(image)
        #
        # top, right, bottom, left = target_face_locations[0]
        # cv2.imshow('target face', target[top: bottom, left: right])
        # cv2.waitKey(0)
        # top, right, bottom, left = image_face_locations[0]
        # cv2.imshow('imgage face', image[top: bottom, left: right])
        # cv2.waitKey(0)

        results = face_recognition.compare_faces([target_face_encoding], image_face_encoding)
        return results[0]
'''

processor = ImageProcessor()
listPixels=processor.generate("C:\\Users\\Catalin\\Desktop\\facultate\\Semestru_5\\MIRPR\\laborator\\aplicatie-git\\New folder\\mirpr-calculafectiv\\app\\server\\agents\\test_images\\putin.png", True)

# img1=imageProcessor.generate("C:\\Users\\Bubu\\mirpr-2020-21\\app\\server\\agents\\Eu.jpg",True)
# img2=imageProcessor.generate(image_path="C:\\Users\\Bubu\\mirpr-2020-21\\app\\user_client\\Connection\\marius.jpeg",show_result=True)
# img2=imageProcessor.generate(image_path="C:\\Users\\Bubu\\mirpr-2020-21\\app\\user_client\\Connection\\img.png",show_result=True)
# while True:
#     cv2.imshow("image",img1)
#     cv2.imshow("image2",img2)
#     if cv2.waitKey(1)& 0xFF == ord('x'):
#          break
# img1 = cv2.imread('D:\\FACULTATE -----FMI\\semestrul 5\\MIRPR\\project\\mirpr-calculafectiv\\app\\user_client\\Connection\\img.png')
# img2 = cv2.imread('D:\\FACULTATE -----FMI\\semestrul 5\\MIRPR\\project\\mirpr-calculafectiv\\app\\user_client\\Connection\\marius.jpg')
# res = imageProcessor.compare(img1, img2)
# print(res)
# path="D:\\altele\\Facultate\\MIRPR\\db\\databrary-30\\databrary30-LoBue-Thrasher-The_Child_Affective_Facial\\sessions\\6282\\10960-sad_F-AA-03.jpg"
# processor = ImageProcessor()
# listPixels=processor.generate(path, True)
