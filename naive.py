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
from utils import *
from sklearn.metrics import confusion_matrix,f1_score
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()


def getBigram(review):
        listt = review.split()
        listt = [lem.lemmatize(i) for i in listt]
        g = nltk.bigrams(review.split())
        lis =[]
        for i in g:
                temp =i[0]
                for j in range(1,len(i)):
                        temp += ' '+i[j]
                lis.append(temp)
        return lis 

def randclassify(review,stars,likehood,p_classs,n_vlaues,vocabsize):
        return random.randint(1,5)
def maxclassify (review,stars,likehood,p_classs,n_vlaues,vocabsize):
        return max(stars)
def classifywithstem(review,stars,likehood,p_classs,n_vlaues,vocabsize):
        posterior_max = -1e7
        ret_class = 0
        review = getBigram(review)
        review = getStemmedDocuments(review)
        
        for i in range(1,6):
                p_class = p_classs[i-1]
                p =0
                n = n_vlaues[i-1]
                for word in review:
                        if word !=[] and len(word[0]) >1:
                                p = p + math.log((likehood[i][word[0]]+1) / (n+vocabsize))
                p = p + math.log(p_class)
                if p> posterior_max:
                        posterior_max =p
                        ret_class = i
        return ret_class

def classifywoutstem(review,stars,likehood,p_classs,n_vlaues,vocabsize):
        posterior_max = -1e7
        ret_class = 0
        review = getBigram(review)  
        for i in range(1,6):
                p_class = p_classs[i-1]
                p =0
                n = n_vlaues[i-1]
                for word in review:
                        if len(word)>1:
                                p = p + math.log((likehood[i][word]+1) / (n+vocabsize))
                p = p + math.log(p_class)
                if p> posterior_max:
                        posterior_max =p
                        ret_class = i
        return ret_class

def classifystem(review,stars,likehood,p_classs,n_vlaues,vocabsize):
        posterior_max = -1e7
        ret_class = 0
        review = getStemmedDocuments(review)
        for i in range(1,6):
                p_class = p_classs[i-1]
                p =0
                n = n_vlaues[i-1]
                for word in review:
                        if len(word)>1:
                                p = p + math.log((likehood[i][word]+1) / (n+vocabsize))
                p = p + math.log(p_class)
                if p> posterior_max:
                        posterior_max =p
                        ret_class = i
        return ret_class
def classify(review,stars,likehood,p_classs,n_vlaues,vocabsize):
        posterior_max = -1e7
        ret_class = 0
        for i in range(1,6):
                p_class = p_classs[i-1]
                p =0
                n = n_vlaues[i-1]
                for word in review.split(' '):
                        if len(word)>1:
                                p = p + math.log((likehood[i][word]+1) / (n+vocabsize))
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
def mains(st):
        f0 = sys.argv[1]  
        f1 = sys.argv[2]   
        partnum = sys.argv[3]   
        correct =0
        total =0
        star_predicted =[]
        star_actual =[]
        likehood = defaultdict(Counter)
        stars = Counter()              
        I=0
        read = json_reader(f0)
        for row in read:
                review = row['text'].strip().lower()
                review = re.sub(r'[^\w\s]','',review)
                review = re.sub(r'[^\w\s]','',review)
                review = re.sub('\r?\n', ' ',review)
                if(partnum =="d" or partnum =="e0" or partnum =="e1"):
                        I+=1
                        if(I%10000==0):
                                print("running ",I)
                        
                        if partnum =="d" or partnum =="e0":    
                                if partnum =="e0":
                                        review = getBigram(review)
                                # print(review)
                                if(type(review)==str):
                                        review = review.split()
                                review = getStemmedDocuments(review)
                                # print(review)
                                stars[int(float(row['stars']))] +=1
                                for word in review:
                                        if word !=[] and len(word[0]) >1:

                                                likehood[int(float(row['stars']))][word[0]] +=1
                        elif partnum =="e1":
                                stars[int(float(row['stars']))] +=1
                                review = getBigram(review)
                                for word in review:
                                        if len(word) >1:
                                                likehood[int(float(row['stars']))][word] +=1
                else: 
                        stars[int(float(row['stars']))] +=1
                        review = review.split()
                        for word in review:
                                if len(word) >1:
                                        likehood[int(float(row['stars']))][word] +=1
        print("here")
        vocabsize =len(likehood[1]+likehood[2]+likehood[3]+likehood[4]+likehood[5])
        print(vocabsize)

        print(time.time()-st)
        n_vlaues =[1]*5
        for i in range(1,6):
                n_vlaues[i-1] = float(sum(likehood[i].values()))
        p_class =[1]*5
        for i in range(1,6):
                p_class[i-1] = float(stars[i])/sum(stars.values())
        
        read = json_reader(f1)
        for row in read:
                review = row['text'].strip().lower()
                review = re.sub(r'[^\w\s]','',review)
                review = re.sub(r'[^\w\s]','',review)
                review = re.sub('\r?\n', ' ',review)
                total = total+1
                if(partnum =="a"):
                        prediction = classify(review,stars,likehood,p_class,n_vlaues,vocabsize)
                if(partnum =="b"):
                        prediction = randclassify(review,stars,likehood,p_class,n_vlaues,vocabsize)
                if(partnum =="b1"):
                        prediction = maxclassify(review,stars,likehood,p_class,n_vlaues,vocabsize)
                if(partnum =="d"):
                        # print("Here")
                        prediction = classifystem(review,stars,likehood,p_class,n_vlaues,vocabsize)
                if(partnum =="e0"):
                        prediction = classifywithstem(review,stars,likehood,p_class,n_vlaues,vocabsize)
                if(partnum=="e1"):
                        prediction = classifywoutstem(review,stars,likehood,p_class,n_vlaues,vocabsize)

                star_predicted.append(prediction)
                star_actual.append(int(float(row['stars'])))
                # print( prediction, int(float(row['stars'])))
                if prediction == int(float(row['stars'])) :
                        correct = correct+1
                if total%10000 ==0:
                        print("acc :",float(correct)/total)
                                                                        
        print(float(correct)/total)
        print(time.time()-st)                        
        confusion_matrix_draw(star_actual,star_predicted)
        if(partnum =="a"):
                f1 = f1_score(star_actual,star_predicted, average= None)
                print("f1 score: ",f1)
                f1_macro = f1_score(star_actual,star_predicted, average= 'macro')
                print(f1_macro)

        
if __name__ == '__main__':
        st = time.time()
        mains(st)
        print("there",time.time()-st)
