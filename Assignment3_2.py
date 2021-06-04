#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:13:44 2019

@author: dr.junk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

#Gaussian Value Calculation
def normalization(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
#Convolution Calculation
def Convolve(img, k):
    
    if len(img.shape) == 3:
        img = cv2.imread(img, 0)
    k = np.flip(np.flip(k,1),0)
    img_row, img_col = img.shape
    k_row, k_col = k.shape

    output = np.zeros(img.shape)

    pad_height = int((k_row - 1) / 2)
    pad_width = int((k_col - 1) / 2)

    padded_image = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img

    for row in range(img_row):
        for col in range(img_col):
            output[row, col] = np.sum(k * padded_image[row:row + k_row, col:col + k_col])

    
    return output
#Sobel Edge detection function
def EdgeDetector(image, filter):
    Ix = Convolve(image, filter)
    Iy = Convolve(image, np.flip(filter.T, axis=0))

    
    return Ix,Iy

#Harris Corner Detection
def HC(image, filter):
     
    Ix,Iy = EdgeDetector(image, filter)
    Ixx=np.square(Ix)
    Iyy=np.square(Iy)
    Ixy=Ix * Iy    
    OneFilter=np.ones((3,3))

    Sxx=Convolve(Ixx,OneFilter)
    Sxy=Convolve(Ixy,OneFilter)
    Syy=Convolve(Iyy,OneFilter)
    
    R=( (Sxx*Syy) - (Sxy**2)) - (0.04* (Sxx-Syy)**2)
    
    corner_eigen=[]
    edge_eigen=[]
    flat_eigen=[]
    eigenMatrix=np.zeros((Ix.shape[0],Ix.shape[1],2))
    regionMatrix=np.zeros((Ix.shape[0],Ix.shape[1]))
    
    for i in range(Ix.shape[0]):
        for j in range(Ix.shape[1]):
            Hessian = np.array([[Sxx[i,j],Sxy[i,j]],[Sxy[i,j],Syy[i,j]]])
            matrix = np.linalg.det(Hessian)
            Htrace = np.trace(Hessian)
            CH = matrix - 0.04*(Htrace**2)
            
            eigen = np.linalg.eigvals(Hessian)
            eigenMatrix[i,j] = eigen
            if CH> 10e8:
                regionMatrix[i,j] = 0
                corner_eigen.append(eigen)
            elif CH<0:
                regionMatrix[i,j] =1
                edge_eigen.append(eigen)
            else:
                regionMatrix[i,j] =2
                flat_eigen.append(eigen)
            
    
    return R,eigenMatrix,regionMatrix,corner_eigen,edge_eigen,flat_eigen

#Gaussian Kernel Creation
def gausskernel(size, sigma=1):
    kernel_1D_matrix = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D_matrix[i] = normalization(kernel_1D_matrix[i], 0, sigma)
    kernel_2D_matrix = np.outer(kernel_1D_matrix.T, kernel_1D_matrix.T)
    kernel_2D_matrix *= 1.0 / kernel_2D_matrix.max()
    return kernel_2D_matrix

def gaussian_blur_filter(image, kernel_size):
    kernel = gausskernel(kernel_size, sigma=math.sqrt(kernel_size))
    return Convolve(image, kernel)

# Calculating LDF
def LDF(featureVector, fcov, meanOfCat):
    gx = np.zeros((10))

    convarianceMatrixInverse = np.linalg.inv(fcov)

    for i in range(cat_no):
        hi = np.dot(convarianceMatrixInverse, meanOfCat[i])
        md = np.dot(np.transpose(featureVector), hi) - 0.5 * np.dot(np.transpose(meanOfCat[i]), hi)
        gx[i] = md[0][0]

    pred = np.argmax(gx)
    return pred

# Image Classification.
def Classify(trainSet, fcov, meanOfCat):
    confusionMatrix = np.zeros((10, 10))
    for index, i in enumerate(trainSet):
        for j in i:
            x = LDF(j.reshape((20, 1)), fcov, meanOfCat)
            confusionMatrix[index, x] += 1
    return confusionMatrix

#Extracting Feature Vector
def featureMatrixCalculation(filename):
    df = pd.read_csv(filename)
    digits = np.arange(10)
    featureMatrix = [[] for i in range(10)]
    for i in digits:
        matrix = []
        index = 0
        for j in df.loc[df.digit == i].values:
            if (index != 50):
                if (j[2] < 10):
                    index += 1
                    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                    image = cv2.imread("data/" + str(i) + "/" + str(j[0]))
                    image = cv2.imread(image, 0)
                    image = gaussian_blur_filter(image, 3)
                    R = HC(image, filter)
                    #to get 20x1 matrix
                    R = np.sort(R.flatten())
                    R = np.concatenate([R[0:10], R[-10:]])
                    featureMatrix[i].append(np.array(R).reshape((20, 1)))
    return np.array(featureMatrix)

def meanCalculation(cat_no,trainSet):
    meanOfCat = np.zeros((cat_no, nfeatures, 1))
    for i in range(cat_no):
        meanOfCat[i] = np.average(trainSet[i], axis=0)
    return meanOfCat

def cov(cat_no,nsamples,trainSet):
    cov_matrix = np.zeros((nfeatures, nfeatures))
    for i in range(cat_no):
        for j in range(nsamples):
            vec = np.array(trainSet[i][j] - totalAvg)
            cov = np.dot(vec, np.transpose(vec))
            cov_matrix += cov
    cov_matrix = cov_matrix / (nsamples * cat_no)
    return cov_matrix



    trainSet = featureMatrixCalculation("DigitDataset/digitTrain.csv")
    testSet = featureMatrixCalculation("DigitDataset/digitTest.csv")


    cat_no = trainSet.shape[0]

    nfeatures = trainSet.shape[2]

    nsamples = trainSet.shape[1]

    meanOfCat = meanCalculation(cat_no,trainSet)

    totalAvg = np.average(np.array(meanOfCat), axis=0)

    cov_matrix=cov(cat_no,nsamples,trainSet)

    trainResult = Classify(trainSet, cov_matrix, meanOfCat)
    testResult = Classify(testSet, cov_matrix, meanOfCat)

    print (testResult)

    #Accuracy Calculation
    trianAcc = np.sum(trainResult * np.eye(10)) / 500
    testAcc = np.sum(testResult * np.eye(10)) / 500


    for i in range(len(meanOfCat)):
        plt.plot(meanOfCat[i],label="class %d"%(i,))
    plt.title("Mean Vector")
    plt.legend(loc='best')
    plt.show()

