#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:17:51 2019

@author: dr.junk
"""
from math import floor,sqrt
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import cv2



#Creating Gaussian kernel
def gausskernel(len=3, nsig=3):
    interval = (2*nsig+1.)/(len)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., len+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


#Convolution
def convolve(image,mask,pad=True):
    if len(image.shape)==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask_shape,_=mask.shape
    mask_flipped=np.flip(np.flip(mask,1),0)
    result=np.zeros(np.array(image.shape)+np.array((mask_shape-1,mask_shape-1)))

    #Padding
    image_padded=result.copy()
    offset=floor(mask_shape/2)
    image_padded[offset:image_padded.shape[0]-offset,offset:image_padded.shape[1]-offset]=image


    for i in range(offset,result.shape[0]-offset):
        for j in range(offset,result.shape[1]-offset):
            result[i,j]=np.sum(mask_flipped*image_padded[i-offset:i-offset+mask_shape,j-offset:j-offset+mask_shape])
    if pad==False:
        return(result[offset:image_padded.shape[0]-offset,offset:image_padded.shape[1]-offset])
    else:
        return(result)

#Edge Detector Function
def edge_detector(mask,image,pad=False):
    
    #creating mask for x and y
    m_x=mask
    m_y=np.flip(np.rot90(mask),0)

    #getting gradients in x nad y directions
    grad_x=convolve(image,mask=m_x,pad=pad)
    grad_y=convolve(image,mask=m_y,pad=pad)

    grad_x=grad_x/grad_x.max()*255.0
    grad_y=grad_y/grad_y.max()*255.0

    return(grad_x,grad_y)


#Harris Corner Detection
def HC(image,mask,gaus=True,window_size=3,k=0.05):
#Finding x and y gradient
    I_x,I_y = edge_detector(mask=mask,image=image)
    mask_shape = mask.shape[0]
    #Finding second order gradients
    I_xx=I_x**2
    I_xy=I_x*I_y
    I_yy=I_y**2

    if gaus == True:
        window=gausskernel(window_size,3)
    else:
        window=np.ones((window_size,window_size))

    sum_xx=convolve(mask=window,image=I_xx,pad=False)
    sum_yy=convolve(mask=window,image=I_yy,pad=False)
    sum_xy=convolve(mask=window,image=I_xy,pad=False)

    #Determinant and Trace
    det=(sum_xx*sum_yy)-(sum_xy**2)
    trace=sum_xx-sum_yy
    R=det-k*(trace**2)
    vec_values=[]
    vec_val=[]

    for ix,i in enumerate(det):
        for iy,j in enumerate(i):
            hessian = np.array([[sum_xx[ix,iy],sum_xy[ix,iy]],[sum_xy[ix,iy], sum_yy[ix,iy]]])
            l1,l2 = np.linalg.eigvals(hessian) 
            vec_values.append(np.array([l1,l2]))
            vec_val.append(np.array([l1,l2,R[ix,iy]]))

    return(R,det,trace,np.array(vec_values),np.array(vec_val))

#Getting Flat Edge and Corner feature values
def get_regions(f_vect,l=.1):
    x_f = f_vect[:,0]
    y_f = f_vect[:,1]

    #Normalising X and Y component 
    x_norm = (x_f-min(x_f))/(max(x_f)-min(x_f))
    y_norm = (y_f-min(y_f))/(max(y_f)-min(x_f))

    #Finding Feature Vectors
    x_bool = x_norm>l
    y_bool = y_norm>l
    f_bool = y_norm>=0

    #Feature vectors of Flat, Corner and Edges
    corner_bool = x_bool*y_bool
    f_corner = f_vect[corner_bool,:]

    edge_bool = np.logical_and((x_bool+y_bool), np.logical_not(corner_bool)) 
    f_edge = f_vect[edge_bool,:]

    flat_bool = np.logical_and(f_bool, np.logical_not((x_bool+y_bool)))
    f_flat = f_vect[flat_bool,:]

    return([f_corner,f_edge,f_flat])


#Getting Flat Edge and Corner feature values
def get_regionsR(f_vectR,l=1,adj=10,fine_thresh =False):
    
    R=f_vectR[:,2]
    unique, counts = np.unique(R, return_counts=True)

    _,bin_edge = np.histogram(R,bins=50)
    loc = np.where(bin_edge[bin_edge <0 ])[0][:-l]


    if fine_thresh == True:
        corner_thresh = bin_edge[-(loc[-1]+adj)]
        edge_thresh = bin_edge[loc[-1]]
    else:

        corner_thresh = (np.average(R[R>0]) + bin_edge[-(loc[-1]+adj)])/2.0 
        edge_thresh = np.average(R[R<0]) 


    f_corner,f_edge,f_flat  = [],[],[]

    for i in f_vectR:
        if i[2] >= corner_thresh:
            f_corner.append(i[:2])
        elif i[2] <= edge_thresh:
            f_edge.append(i[:2])
        else:
            f_flat.append(i[:2])
    return([np.array(f_corner),np.array(f_edge),np.array(f_flat),corner_thresh,edge_thresh])

def plot(f,e,c,axis):
    data = (f,e,c)
    colors = ("orange","green", "red")
    groups = ("flat","edge", "corner")

    # Graph
    for data, color, group in zip(data, colors, groups):
        x, y = data[:,0],data[:,1]
        axis.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
        axis.legend(loc=2)
    return(axis)

def _split_e(e):
    e_l,e_r=[],[]
    for i in e:
        if i[0]>i[1]:
            e_r.append(i)
        else:
            e_l.append(i)
    return(np.array(e_l),np.array(e_r))

#Finding Covariance Matrix
def _cov(val,val_mean):
    cov_matrix = []

    for value in val:
        vec = np.array([np.array(value - val_mean)])
        cov = np.dot(np.transpose(vec),vec)
        cov_matrix.append(cov)

    covariance = np.average(np.array(cov_matrix),axis=0)

    return(covariance)

def _fisher_space(val,eigen_vector):
    fisher_space=[]
    for i in val:
        fisher_space.append(np.dot(np.transpose(eigen_vector),np.transpose(np.array([i]))))
    return(np.array(fisher_space))


def get_fisher(c,e,f):

    e_l,e_r = _split_e(e)
    classes = [c,e_l,e_r,f]

    c_mean = np.average(c,axis=0)
    e_l_mean = np.average(e_l,axis=0)
    e_r_mean = np.average(e_r,axis=0)
    f_mean = np.average(f,axis=0)
    class_mean = [np.array([x]).reshape((2,1)) for x in [c_mean,e_l_mean,e_r_mean,f_mean]]
    overall_mean = np.average(np.array(class_mean),axis=0).reshape((2,1))

    c_cov = _cov(c,c_mean)
    e_l_cov = _cov(e_l,e_l_mean)
    e_r_cov = _cov(e_r,e_r_mean)
    f_cov = _cov(f,f_mean)

#Storing them in a list
    class_cov = [c_cov,e_l_cov,e_r_cov,f_cov]

    B = np.zeros_like(c_cov)
    A = np.zeros_like(c_cov)
    for mean,cov in zip(class_mean,class_cov):
        vec = np.array(mean - overall_mean)
        B += np.dot(vec,np.transpose(vec))
        A += cov
    A_inv = np.linalg.inv(A)
    h = np.dot(A_inv,B)

    eigen_value, eigen_vector = np.linalg.eig(h)



    fisher_class = []
    for val in classes:
        fisher_class.append(_fisher_space(val,eigen_vector))

    all_data = np.vstack(fisher_class)
    data =[fisher_class[3],np.vstack([fisher_class[1],fisher_class[2]]),fisher_class[0]]

    return(data,all_data,eigen_vector,[np.dot(np.dot(np.transpose(eigen_vector),i),eigen_vector) for i in class_cov],
        [np.dot(np.transpose(eigen_vector),i) for i in class_mean],np.dot(np.transpose(eigen_vector),overall_mean))

def plot_mean_distributions(fisher_mean,fisher_overall_mean,c,e,f,new_data):

    fisher_mean.append(fisher_overall_mean)
    fisher = np.array([i.reshape((2)) for i in fisher_mean]) 

    e_l,e_r = _split_e(e)
    c_mean = np.average(c,axis=0)
    e_l_mean = np.average(e_l,axis=0)
    e_r_mean = np.average(e_r,axis=0)
    f_mean = np.average(f,axis=0)
    class_mean = [np.array([x]).reshape((2,1)) for x in [c_mean,e_l_mean,e_r_mean,f_mean]]
    overall_mean = np.average(np.array(class_mean),axis=0).reshape((2,1))
    actual_mean = np.array([np.array([x]).reshape((2)) for x in [c_mean,e_l_mean,e_r_mean,f_mean,overall_mean]])

    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    ax3 = plot(new_data[0],new_data[1],new_data[2],ax3)
    ax3.set_title('Fisher Space Distribution')
    ax4 = plot(f,e,c,ax4)
    ax4.set_title('Actual Distribution')

    label = ['corner','edge_left','edge_right','flat','overall']
    color = ['r','g','g','b','c']
    for x,y,l,c in zip(fisher[:,0],fisher[:,1], label,color): 
        ax1.scatter(x,y,label = l, c = c) 
        ax1.legend(loc=2)
        ax1.set_title(label = 'Fisher mean distribution')
    
    for x,y,l,c in zip(actual_mean[:,0],actual_mean[:,1],label,color):
        ax2.scatter(x,y,label = l, c = c) 
        ax2.legend(loc=2)
        ax2.set_title(label = 'Actual mean distribution')
    plt.show()
    return(plt)

def actual_classifier(co,ed,fl):
    corner = np.array([np.append(i,[0],axis=0) for i in co])
    edge = np.array([np.append(i,[1],axis=0) for i in ed])
    flat = np.array([np.append(i,[2],axis=0) for i in fl])
    features = np.vstack([corner, edge, flat])
    return(features)

def classify(val,e_vector,f,f_mean):
    classes = ['corner','edge','edge','flat']
    val = np.array(val).reshape((2,1))
    mahalnobis_dist = np.zeros(4)
    feature = np.dot(np.transpose(e_vector),val)
    for i,(mean,cov_inv) in enumerate(zip(f_mean,f)):
        vec = feature-mean
        mahalnobis_dist[i] = np.dot(np.dot(np.transpose(vec),cov_inv),vec)
    feature_class=np.argmin(mahalnobis_dist)

    return(classes[feature_class],mahalnobis_dist)





#Loading Images
image1=cv2.imread('input_hcd1.jpg')
mask=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

#Harris corner response recorded
harris_response,det,trace,f_vect,f_vectR =HC(image1,mask,gaus=False)


#Threshold values 
c,e,f,corner_thresh,edge_thresh = get_regionsR(f_vectR)

#Actual Classifier dataset
feature = actual_classifier(c,e,f)

#Distribution Plot
r_based  = plot(f,e,c,plt)


#Mean variation Plotting
new_data,fisher_space,eigen,fisher_cov,fisher_mean,fisher_overall_mean = get_fisher(c,e,f)
plt=plot_mean_distributions(fisher_mean,fisher_overall_mean,c,e,f,new_data)
plt.show()



# Classfiying based on the mahalnobis distance
classes = {'corner':0,'edge':1,'flat':2}
fisher_cov_inv = [np.linalg.inv(x) for x in fisher_cov]
cm = np.zeros((len(classes),len(classes))) 
for i in feature:
    #Fisher space 2 * 1 vector
    val = i[:2]
    detected_class = classify(val,eigen,fisher_cov_inv,fisher_mean)  
    actual_class = int(i[2])

    predicted_class = classes[detected_class[0]]
    cm[actual_class,predicted_class] += 1
