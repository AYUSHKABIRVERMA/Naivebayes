import random
import numpy as np 
import math
import string
import csv
import re
import json
import sys
from collections import Counter,defaultdict
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
      
def randclassify(review,likehood):
        return random.randint(1,5)
def maxclassify (review,stars,likehood):
        return max(stars)
def classify(review,stars,likehood):
        posterior_max = -1e5
        ret_class = 0
        for i in range(1,6):
                p_class = float(stars[i])/sum(stars.values())
                p =0
                n = float(sum(likehood[i].values()))
                for word in review.split(' '):
                        if len(word)>1:
                                p = p + math.log((likehood[i][word]+1) / (n+len(likehood[i])))
                p = p + math.log(p_class)
                if p> posterior_max:
                        posterior_max =p
                        ret_class = i
        return ret_class
def confusion_matrix_draw(star_actual,star_predicted):
        cm = confusion_matrix(star_actual,star_predicted)
        plt.imshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.set_cmap('Blues')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
def mains():
        f0 = sys.argv[1]  
        f1 = sys.argv[2]      
        correct =0
        total =0
        star_predicted =[]
        star_actual =[]
        likehood = defaultdict(Counter)
        stars = Counter()              
        with open(f0,'r') as a :
                read = csv.reader(a)
                for row in read:
                        if row[3]!="text" :
                                review = row[3].strip().lower()
                                review = re.sub(r'[^\w\s]','',review)
                                review = re.sub(r'[^\w\s]','',review)
                                review = re.sub('\r?\n', ' ',review)
                                stars[int(float(row[5]))] +=1
                                for word in review.split(' '):
                                        if len(word) >1:
                                                likehood[int(float(row[5]))][word] +=1

        print("here")
        with open(f1,'r') as f:
                read = csv.reader(f)
                for row in read:
                        if row[3]!="text":
                                review = row[3].strip().lower()
                                review = re.sub(r'[^\w\s]','',review)
                                review = re.sub(r'[^\w\s]','',review)
                                review = re.sub('\r?\n', ' ',review)
                                total = total+1
                                prediction = classify(review,stars,likehood)
                                star_predicted.append(prediction)
                                star_actual.append(int(float(row[5])))
                                if prediction == int(float(row[5])) :
                                        correct = correct+1
                                        # if correct% 100 ==0 :
                                        #         print(float(correct)/total)
                                        if correct> 100:
                                                break
        print(float(correct)/total)                        
        confusion_matrix_draw(star_actual,star_predicted)
        
if __name__ == '__main__':
        st = time.time()
        mains()
        print("there")
        print(float(time.time()- st)/60)
