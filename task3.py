# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:52:05 2018

@author: Aniruddha Sinha
UB Person Number = 50289428
UBIT = asinha6@buffalo.edu
"""

UBIT = '<asinha6>'; import numpy as np; np.random.seed(sum([ord(c) for c in UBIT]))

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import math
import time
import csv
from matplotlib.patches import Ellipse

def classifyPoints(X, Mu, Cov, pdf):
    C1 = []
    C2 = []
    C3 = []
    classification_vector = []
    
    pdf1 = []
    pdf2 = []
    pdf3 = []
    
    if pdf:
        for i in range(len(X)):
            L2_C1 = multivariate_normal.pdf(X[i], mean = Mu[0,:], cov=Cov)
            pdf1.append(L2_C1)
            
            L2_C2 = multivariate_normal.pdf(X[i], mean = Mu[1,:], cov=Cov)
            pdf2.append(L2_C2)
            
            L2_C3 = multivariate_normal.pdf(X[i], mean = Mu[2,:], cov=Cov)
            pdf3.append(L2_C3)
            
            if max(L2_C1,L2_C2,L2_C3) == L2_C1:
                C1.append(X[i])
                classification_vector.append("Mu1")
            elif max(L2_C1,L2_C2,L2_C3) == L2_C2:
                C2.append(X[i])
                classification_vector.append("Mu2")
            else:
#                print("L2_C3", L2_C3)
                C3.append(X[i])
                classification_vector.append("Mu3")
    else:
        for i in range(len(X)):
            L2_C1 = math.sqrt((X[i][0] - Mu[0][0])**2 + (X[i][1] - Mu[0][1])**2)
            
            L2_C2 = math.sqrt((X[i][0] - Mu[1][0])**2 + (X[i][1] - Mu[1][1])**2)
            
            L2_C3 = math.sqrt((X[i][0] - Mu[2][0])**2 + (X[i][1] - Mu[2][1])**2)
            
            if min(L2_C1,L2_C2,L2_C3) == L2_C1:
                C1.append([ X[i][0], X[i][1] ])
                classification_vector.append("Mu1")
            elif min(L2_C1,L2_C2,L2_C3) == L2_C2:
                C2.append([ X[i][0], X[i][1] ])
                classification_vector.append("Mu2")
            else:
                C3.append([ X[i][0], X[i][1] ])
                classification_vector.append("Mu3")
    
#    print('\n\nPDF for Mu1:', pdf1, '\n\nPDF for Mu2:', pdf2, '\n\nPDF for Mu3:', pdf3)    
    return C1, C2, C3, classification_vector


def plotWithText(Arr):
   for i in range(len(Arr)):
    s = '('+str(Arr[i][0])+','+str(Arr[i][1])+')'
    plt.text(Arr[i][0], Arr[i][1], s)

def plotMu(Mu):
    plt.scatter(Mu[0,0], Mu[0,1], c = 'r', marker="o", edgecolors=None, facecolor='none')
    plt.scatter(Mu[1,0], Mu[1,1], c = 'g', marker="o", edgecolors=None, facecolor= 'none')
    plt.scatter(Mu[2,0], Mu[2,1], c = 'b', marker="o", edgecolors= None, facecolor= 'none')

def updateMu(classifiedPoints):
    
    xSum= 0 
    ySum = 0
    for i in range(len(classifiedPoints)):
        xSum += classifiedPoints[i][0]
        ySum += classifiedPoints[i][1]
    
    xAvg = xSum/len(classifiedPoints)
    xAvg = round(xAvg,1)
    
    yAvg = ySum/len(classifiedPoints)
    yAvg = round(yAvg,1)
    
    return [xAvg,yAvg]

def plotClassified(C1,C2,C3,Mu, filename):
    
    np_C1 = np.array(C1)
    np_C2 = np.array(C2)
    np_C3 = np.array(C3)
    
    plt.figure()
    plotMu(Mu)
    plotWithText(Mu)
    
    if np_C1.shape[0]>0:  
        plt.scatter(np_C1[:,0], np_C1[:,1], c='r', marker="^", edgecolors='r', facecolor='none')
        plotWithText(np_C1)
    
    if np_C2.shape[0]>0:
        plt.scatter(np_C2[:,0], np_C2[:,1], c='g', marker="^", edgecolors='g', facecolor='none')
        plotWithText(np_C2)
    
    if np_C3.shape[0]>0:
        plt.scatter(np_C3[:,0], np_C3[:,1], c='b', marker="^", edgecolors='b', facecolor='none')
        plotWithText(np_C3)
    
    plt.savefig('./results/'+filename+'.jpg')
    
def part1(X, Mu):
    
    C1, C2, C3, classification_vector = classifyPoints(X, Mu, None, False)
    print('\n\nFirst cluster for Mu1: ', C1)
    print('\n\nSecond cluster for Mu2:', C2)
    print('\n\nThird cluster for Mu3:', C3)
    
    print('\n\nClassification Vector is:\n', classification_vector)
    
    plotClassified(C1,C2,C3,Mu,'task3_iter1_a')
    
def part2(X,Mu):
    
    C1, C2, C3, classification_vector = classifyPoints(X,Mu, None, False)
    
    updatedMu = []

    updatedMu.append(updateMu(C1))
    updatedMu.append(updateMu(C2))
    updatedMu.append(updateMu(C3))
    updatedMu = np.array(updatedMu)
    
    plt.figure()   
    plt.scatter(X[:,0], X[:,1], marker="^", edgecolor = 'b', facecolor='None')
    plotWithText(X)
    plotMu(updatedMu)
    plotWithText(updatedMu)
    plt.savefig('./results/task3_iter1_b.jpg')

    print('\n\nUpdated Mu after iteration 1: \n', updatedMu)
    
    return updatedMu

def part3(X,updatedMu):    
    
    C1, C2, C3, classification_vector = classifyPoints(X,updatedMu, None, False)

    print('\n\nUpdated cluster for Mu1: ', C1)
    print('\n\nUpdated cluster for Mu2:', C2)
    print('\n\nUpdated cluster for Mu3:', C3)
    
    print('\n\nUpdated Classification Vector is:\n', classification_vector)
    
    plotClassified(C1,C2,C3,updatedMu, 'task3_iter2_a')
    
    updated2Mu = []

    updated2Mu.append(updateMu(C1))
    updated2Mu.append(updateMu(C2))
    updated2Mu.append(updateMu(C3))
    updated2Mu = np.array(updated2Mu)
    
    plt.figure()   
    plt.scatter(X[:,0], X[:,1], marker="^", edgecolor = 'b', facecolor='None')
    plotWithText(X)
    plotMu(updated2Mu)
    plotWithText(updated2Mu)
    plt.savefig('./results/task3_iter2_b.jpg')

    print('\n\nUpdated Mu after iteration 2: \n', updated2Mu)

def updateMuBaboon(classifiedPoints):
    
    xSum= 0 
    ySum = 0
    zSum = 0
    for i in classifiedPoints:
        xSum += i[0]
        ySum += i[1]
        zSum += i[2]
    
    xAvg = xSum/len(classifiedPoints)
    xAvg = round(xAvg)
    
    yAvg = ySum/len(classifiedPoints)
    yAvg = round(yAvg)
    
    zAvg = zSum/len(classifiedPoints)
    zAvg = round(zAvg)
    
    return xAvg,yAvg,zAvg

def part4(baboon):
    
    h,w,c = baboon.shape
    reshapedBaboon = baboon.reshape(-1,3)
    
    Mu = [3,5,10,20]
    
    for m in Mu:
        Mu_baboon = reshapedBaboon[5:m+5]
        muLen = Mu_baboon.shape[0]
        muLen2 = Mu_baboon.shape[1]
        
        muPixels = []
        
        for K in range(10):
            
            classification_vector = []
            C_matrix = []
            
            for num_clust in range(m):
                C_matrix.append([])
                
            MuList = []
            for pixel in range(reshapedBaboon.shape[0]):
                
                temp = np.zeros((muLen))
                
                for i in range(muLen):
                    tempMu = 0
                    for j in range(muLen2):
                        tempMu += (reshapedBaboon[pixel][j] - Mu_baboon[i][j])**2
                    temp[i] = math.sqrt(tempMu)
                
                temp = temp.tolist()    
                muIndex = temp.index(min(temp))
                MuList.append(muIndex)
                
                C_matrix[muIndex].append(reshapedBaboon[pixel])
                
                classification_vector.append(Mu_baboon[muIndex])
                
            muPixels = []
            for muInd in range(muLen):
                pixelsofThisMu = [i for i, e in enumerate(MuList) if e == muInd]
                muPixels.append(reshapedBaboon[pixelsofThisMu])
            
            Mu_baboon = []
            for i in range(len(muPixels)):    
                Mu_baboon.append(updateMuBaboon(muPixels[i]))
            
            Mu_baboon = np.array(Mu_baboon)
#            print(Mu_baboon, Mu_baboon.shape)
        
            classification_vector = np.array(classification_vector)
            output = classification_vector.reshape(h,w,c)
        global t
        cv2.imwrite("results/task3_baboon_"+str(m)+".jpg",output)
        print('\n\n***************** task3_baboon'+str(m)+'.jpg written to file.***************************')
        print('\n\nTime taken for K='+str(m)+' is ', time.time()-t)


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''' FUNCTIONS FOR BONUS START ''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def getFaithfulData(filePath):
    t = []
    count = 0
    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if count == 0:
                count = 1
                continue
            t.append(np.float32(row))
    return t

def plotClassifiedforBonus(C1,C2,C3,Mu):
    
    plotMu(Mu)
    if C1.shape[0]>0:  
        plt.scatter(C1[:,0], C1[:,1], c='r', marker="o", edgecolors='r', facecolor='none')
    
    if C2.shape[0]>0:
        plt.scatter(C2[:,0], C2[:,1], c='g', marker="o", edgecolors='g', facecolor='none')
        
    if C3.shape[0]>0:
        plt.scatter(C3[:,0], C3[:,1], c='b', marker="o", edgecolors='b', facecolor='none')
        

"""

The functions plotPointsEllipse have been copied from
https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py


"""
def plotPointsEllipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2.5 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def part5_bonus():
    X = np.array([[5.9,3.2], [4.6,2.9], [6.2,2.8], [4.7,3.2], [5.5,4.2], [5.0,3.0], [4.9,3.1], [6.7,3.1], [5.1,3.8], [6.0,3.0]])
    Mu = np.array([[6.2,3.2], [6.6,3.7], [6.5,3.0]])
    Cov = [[0.5,0], [0,0.5]]
    
    C1, C2, C3, classification_vector = classifyPoints(X,Mu, Cov, True)
    
    updatedMu = []
    
    updatedMu.append(updateMu(C1))
    updatedMu.append(updateMu(C2))
    updatedMu.append(updateMu(C3))
    updatedMu = np.array(updatedMu)
    
    print('\nUpdated Mu after 1st iteration on given dataset is:', updatedMu)
    
    faithfulData = np.array(getFaithfulData('data/faithful.csv'))
    print('\n\n',faithfulData.shape, type(faithfulData), faithfulData[1,:])
    
    computeFaithful =  faithfulData[:,1:]
    Mu_faithful = np.array([[4.0, 81], [2.0, 57], [4.0,71]])
    Cov_f = np.array([[1.30, 13.98], [13.98, 184.82]])
    
    print('\n\n################# Starting computation on the faithful data: #######################\n\n')
    print('\n\n         For Faithful Data:          ')      
          
    for i in range(5):
        f_C1, f_C2, f_C3, classification_faith = classifyPoints(computeFaithful, Mu_faithful, Cov_f, True)
        Mu_faithful = []
        
        Mu_faithful.append(updateMu(f_C1))
        Mu_faithful.append(updateMu(f_C2))
        Mu_faithful.append(updateMu(f_C3))
        Mu_faithful = np.array(Mu_faithful)
        
        print('\n\nUpdated Mu after iteration '+str(i+1)+' is:', Mu_faithful)
        
        f_C1 = np.array(f_C1)
        f_C2 = np.array(f_C2)
        f_C3 = np.array(f_C3)
        
        if i==0:
            f_c1 = f_C1.mean(axis=0)
            cov_f1 = Cov_f
            
            f_c2 = f_C2.mean(axis=0)
            cov_f2 = Cov_f
            
            f_c3 = f_C3.mean(axis=0)
            cov_f3 = Cov_f
        else:
            f_c1 = f_C1.mean(axis=0)
            cov_f1 = np.cov(f_C1, rowvar=False)
            
            f_c2 = f_C2.mean(axis=0)
            cov_f2 = np.cov(f_C2, rowvar=False)
            
            f_c3 = f_C3.mean(axis=0)
            cov_f3 = np.cov(f_C3, rowvar=False)
        
        plt.figure()  
        plt.title('Ellipse plots after Iteration'+str(i+1)) 
        plotClassifiedforBonus(f_C1, f_C2, f_C3,Mu_faithful)
        plotPointsEllipse(cov_f1, f_c1, nstd=3, alpha=0.5, color='red')
        plotPointsEllipse(cov_f2, f_c2, nstd=3, alpha=0.5, color='green')
        plotPointsEllipse(cov_f3, f_c3, nstd=3, alpha=0.5, color='blue')    
        plt.savefig('./results/task3_gmm_iter'+str(i+1)+'.jpg')
        plt.show()

def kmeans_implement():
    
    X = np.array([[5.9,3.2], [4.6,2.9], [6.2,2.8], [4.7,3.2], [5.5,4.2], [5.0,3.0], [4.9,3.1], [6.7,3.1], [5.1,3.8], [6.0,3.0]])
    Mu = np.array([[6.2,3.2], [6.6,3.7], [6.5,3.0]])
    
    plt.figure()   
    plt.scatter(X[:,0], X[:,1], marker="^", edgecolor = 'b', facecolor='None')
    plotWithText(X)
    plotMu(Mu)
    plotWithText(Mu)
    plt.savefig('./results/task3_original.jpg')
    
    global t
    ############################ START PART 1 ####################################
    print('\n\n##################### Starting execution of Task 3 part 1 ##########################################')
    part1(X,Mu)
    print('\n\n############### Task 3 part 1 successfully completed.#####################\nTime taken for Task 3 part 1 = ',time.time()-t,' seconds')

    ############################ END PART 1 #######################################
    
    ############################ START PART 2 #####################################
    print('\n\n##################### Starting execution of Task 3 part 2 ##########################################')
    updatedMu = part2(X,Mu)
    print('\n\n############### Task 3 part 2 successfully completed.#####################\nTime taken for Task 3 part 2 = ',time.time()-t,' seconds')

    ####################### END PART 2 ############################################
    
    ####################### START PART 3 ##########################################
    print('\n\n##################### Starting execution of Task 3 part 3 ##########################################')
    part3(X, updatedMu)
    print('\n\n############### Task 3 part 3 successfully completed.#####################\nTime taken for Task 3 part 3 = ',time.time()-t,' seconds')
    ###################### END PART 3 ############################################
    
    ########################## START PART4 #######################################
    print('\n\n##################### Starting execution of Task 3 part 4 ##########################################')
    print('\n\n                 Please wait for 15-20 minutes. This will take some time....                      ')      
    baboon = cv2.imread('data/baboon.jpg')
    part4(baboon)
    print('\n\n############### Task 3 part 4 successfully completed.#####################\nTime taken for Task 3 part 4 = ',time.time()-t,' seconds')
    
    ###################### START BONUS: PART 5 #########################################
    print('\n\n##################Starting execution of Task 3: BONUS ###########################')
    print('\n\n*********** Task 3: Bonus , Part 1, started execution: *********************** ')
    part5_bonus()
    print('\n\n############### Task 3 part 5 - BONUS successfully completed.#####################\nTime taken for Task 3 BONUS:part 5 = ',time.time()-t,' seconds')          
        

if __name__=='__main__':
    try:
        global t
        t = time.time()
        kmeans_implement()
    except:
        pass
             

