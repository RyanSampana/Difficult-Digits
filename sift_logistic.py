# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 01:11:15 2016

@author: francolin
"""

import numpy as np
import scipy.misc
from sklearn import linear_model
import csv
import datetime
import time
from sklearn import metrics
from PIL import Image
import cv2
from sklearn import model_selection
from sklearn import naive_bayes

max_lines = -1
start_time = time.time()


def read_y_data(file_name):
    image_id = []
    y = []
    
    with open(file_name, 'r') as csvfile:
        read = csv.reader(csvfile, delimiter=',')
        num_lines = 0
        next(read)
        for row in read:
            num_lines += 1
            image_id.append(int(row[0]))
            y.append(int(row[1]))
        
            if num_lines > max_lines and max_lines > 0:
                break
    
    return y, image_id


def extract_sift(image):
    sift = cv2.SIFT()
    dense=cv2.FeatureDetector_create("Dense")
    kp = dense.detect(image)
    kp, kd = sift.compute(image, kp)
    return kp, kd

def extract_all(images):
    print "Extracting keypoints"
    kds = []
    kps = []    
    for img in images:
        kp, kd = extract_sift(img)
        kds.append(kd)
        kps.append(kp)
        
    print "Extracted keypoints from %d images" % len(kds)
    return kps, kds
        
    

def k_fold_cv(model,folds,x_data,y_data):
    print("Running CV")
    k_fold = model_selection.KFold(n_splits = folds)
    cv_scores = []
    train_scores = []
    
    for train_index, test_index in k_fold.split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        curr_acc = metrics.accuracy_score(y_test,y_predict)
        cv_scores.append(curr_acc)
        print("CV ACC: " + str(curr_acc))
        y_train_predict = model.predict(x_train)
        train_acc = metrics.accuracy_score(y_train,y_train_predict)
        train_scores.append(train_acc)
        print("Train ACC: " + str(train_acc))
    
    print('\n'+str(cv_scores))
    print("Average CV ACC: "  + str(np.mean(cv_scores)))
    print('\n'+str(train_scores))
    print("Average Train ACC: "  + str(np.mean(train_scores)))


def grid_search_cv(model, folds, parameters, score, x_data, y_data):
    
    grid_search = model_selection.GridSearchCV(model, parameters, cv=folds, scoring=score, verbose=True)
    grid_search.fit(x_data, y_data)
    print'Parameters : '  + str(grid_search.cv_results_['params'])
    print'Scores : ' + str(grid_search.cv_results_['mean_test_score'])
    print'Ranks : ' + str(grid_search.cv_results_['rank_test_score'])
    
    print'\nBest Score : ' + str(grid_search.best_score_)
    print'Best Parameters : ' + str(grid_search.best_params_)
    
    final_model = grid_search.best_estimator_
    print(final_model)
    
    return final_model


if __name__ == "__main__":
    x = np.fromfile('train_x.bin', dtype='uint8')
    x = x.reshape((100000,60,60))

    
    y, image_id = read_y_data('train_y.csv')
    #sift to csv and train/test
    x = x[:]
    y = y[:]
    
    x_data = []
    y_data = []


    kps, kds = extract_all(x)
    print('Time Elapsed: '+ str(datetime.timedelta(seconds=time.time()-start_time))+'\n')
#    print(len(min(kps,key=len)))

    no_keypoint = []
        
    
#    csv_file =  open("dense_sift_features.csv", "w")    
#    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    for i in range(len(kds)):
        if kds[i] == None:
            no_keypoint.append(image_id[i])
            print('empty keypoints')
            print(image_id[i])
        else:
#            wr.writerow(kds[i][0])
            x_data.append(kds[i][0]+kds[i][1]+kds[i][2])
            y_data.append(y[i])
    print(no_keypoint)
#    csv_file.close()
    
    
    
#    max_lines = -1
##    del y[57512]
#    y_data = y
#    x_data = []
#    file_name = 'dense_sift_features.csv'
#    print("Reading Sift csv...")
#    num_lines = 0
#    with open(file_name, 'r') as csvfile:
#        read = csv.reader(csvfile, delimiter=',')
#        for row in read:
#            x_data.append([float(val) for val in row])
#            num_lines += 1
#            if num_lines > max_lines and max_lines > 0:
#                break


    logistic = linear_model.LogisticRegression(solver='sag', n_jobs=3, max_iter=10000)
    
    parameters = {'C':[0.001,0.01,0.1,1,10,100], 'tol':[0.0001,0.001,0.01,0.1]}
#    parameters = {'C':[0.1,1], 'tol':[0.001]}
    scores = 'accuracy'
    folds = 5
    print(len(x_data))
    print(len(y_data))
#    final_model = grid_search_cv(logistic, folds, parameters, scores, np.array(x_data), np.array(y_data))
    k_fold_cv(logistic,5,np.array(x_data),np.array(y_data))
    

    
    print('Time Elapsed: '+ str(datetime.timedelta(seconds=time.time()-start_time))+'\n')
