import sys
import math
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
import pandas as pd 
from scipy.misc import imread 
from sklearn.naive_bayes import GaussianNB



def hough(image):
    src = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    dst = cv2.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    return [cdst,cdstP]
    

def read_img(filename):
    try:
        img = cv2.imread(filename,-1)
        if len(img)>0:
            return [img]
    except Exception as e:
        return -1


def extract_features(image, vector_size=32):
    try:
        # image = imread(image_path, mode="RGB")

        b,g,r = cv2.split(image)       # making it rgb instead of bgr
        image = cv2.merge([r,g,b])
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        # except Exception as e:
        #     print('Error: ', e)
        #     return [None]
        return [dsc]
    except Exception as e:
        return -1


def hough_extract_features():

    print("Started")
    path="DATASET\\TRAIN\\O\\O_"+str(1)+".jpg"

    data = read_img(path)
    if data != -1:
        img = data[0]
        t = hough(img)
        if t != -1:    
            features_array = extract_features(t[0])
            if features_array != -1:
                features = features_array[0]
                X = np.array([features])
                Y = np.array([0])

    for i in range(2,12600):
        path="DATASET\\TRAIN\\O\\O_"+str(i)+".jpg"
        if(i % 500 == 0):
            print(i , " done")
        
        data = read_img(path)
        if data != -1:
            img = data[0]
            t = hough(img)
            if t != -1:    
                features_array = extract_features(t[0])
                if features_array != -1:
                    features = features_array[0]

                    if features[0]!=None:
                        X = np.append(X,[features],axis=0) 
                        Y = np.append(Y,[0],axis=0)
        

    for i in range(1,10000):
        path="DATASET\\TRAIN\\R\\R_"+str(i)+".jpg"
        if(i % 500 == 0):
            print(i," done")
        
        data = read_img(path)
        if data != -1:
            img = data[0]
            t = hough(img)
            if t != -1:    
                features_array = extract_features(t[0])
                if features_array != -1:
                    features = features_array[0]
                    if features[0]!=None:
                        X = np.append(X,[features],axis=0) 
                        Y = np.append(Y,[1],axis=0)

    print(len(X))
    print(len(Y))


    print("Started Testing Data Collection")
    

    path="DATASET\\TEST\\O\\O_"+str(12568)+".jpg"
    data = read_img(path)
    if data != -1:
        img = data[0]
        t = hough(img)
        if t != -1:    
            features_array = extract_features(t[0])
            if features_array != -1:
                features = features_array[0]
                if features[0]!=None:
                    Xtest = np.append(X,[features],axis=0) 
                    Ytest = np.append(Y,[0],axis=0)


    for i in range(12569,14000):
        path="DATASET\\TEST\\O\\O_"+str(i)+".jpg"
        
        if( i % 500 == 0):
            print(i," done")

        data = read_img(path)
        if data != -1:
            img = data[0]
            t = hough(img)
            if t != -1:    
                features_array = extract_features(t[0])
                if features_array != -1:
                    features = features_array[0]
                    if features[0]!=None:
                        Xtest = np.arrays([features]) 
                        Ytest = np.arrays([0])

    for i in range(10000,11112):
        path="DATASET\\TEST\\R\\R_"+str(i)+".jpg"
        
        if(i % 500 == 0):
            print(i," done")

        data = read_img(path)
        if data != -1:
            img = data[0]
            t = hough(img)
            if t != -1:    
                features_array = extract_features(t[0])
                if features_array != -1:
                    features = features_array[0]
                    if features[0]!=None:
                        Xtest = np.append(X,[features],axis=0) 
                        Ytest = np.append(Y,[1],axis=0)

    print(len(Xtest))
    print(len(Ytest))

    return X,Y,Xtest,Ytest


if __name__ == "__main__":

    # filename = 'DATASET\TRAIN\O\O_1.jpg'
    # data = read_img(filename)
    # if data != -1:
    #     img = data[0]
    #     t = hough(img)
    #     if t != -1:    
    #         features_array = extract_features(t[0])
    #         if features_array != -1:
    #             features = features_array[0]
    #             print(len(features))

    # X,Y,Xtest,Ytest = hough_extract_features()

    # gnb = GaussianNB()
    # gnb.fit(X,Y)
    # print("Training acc:",gnb.score(X,Y))
    # print("Testing acc:",gnb.score(Xtest,Ytest))
	path="DATASET\\TRAIN\\O\\O_"+str(122)+".jpg"
	data = read_img(path)
	if data != -1:
	    img = data[0]
	    t = hough(img)
	    if t != -1:    
	        cv2.imshow("image",t[1])
	        cv2.waitKey(0)
	        features_array = extract_features(t[0])
	        if features_array != -1:
	            features = features_array[0]
	            X = np.array([features])
	            Y = np.array([0])

