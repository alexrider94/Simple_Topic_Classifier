import sys
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import math

#########################################################
#필요 변수
#########################################################
lineList = []
categories = ['interest','jobs','money_supply','trade']
cateogory_dict = {}
contentList = []
prob_topic_dict = {}
prob_word_dict={}
testList=[]

#불용어 제거 변수
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update(['would','us','also','the','we\'ve','i'])

#########################################################
# 함수
#########################################################
def stem(word):
    regExp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|ment)?$'
    stem, suffix = re.findall(regExp, word)[0]
    return stem

def getTextFromFile(content,category):
    wordList = textClassify(content)
    for word in wordList:
        if category in cateogory_dict:
            cateogory_dict[category].append(word)

def textClassify(content):
    wordList=[]
    for line in content:
        line = line.lower()
        lineList = line.strip().split(" ")
        for word in lineList:
            if word not in stop_words:
                temp = stem(word)
                temp = temp.replace("\"","").replace(".","").replace(",","")
                wordList.append(temp)
    return wordList

def training(cateogory_dict):
    all_of_length = 0
    dict_topic_length={}
    dict_word_by_topic_result={}
    dict_categoies_result={}
    all_of_words=[]
    word_counted={}
    dict_categories_length={}
    for category in cateogory_dict:
        words = cateogory_dict.get(category)
        length = len(words)
        all_of_length += length
        for w in words:
            all_of_words.append(w)
    
    for category in cateogory_dict:
        words = cateogory_dict.get(category)
        counted_word={}
        dict_categories_length[category]=len(words)
        for w in all_of_words:
            if w not in words:
                counted = 0.3
                counted_word[w]=counted
            else:
                counted= (words.count(w) - (0.3*words.count(w)))
                counted_word[w]=counted
            counted= words.count(w)
        word_counted[category]=counted_word

    for category,word in word_counted.items():
        sum = 0
        c_length = dict_categories_length[category]
        for w,count in word.items():
            pro = count / c_length
            sum += pro
            
            dict_categoies_result[w]=pro
        
        dict_topic_length[category] = c_length/all_of_length
        dict_word_by_topic_result[category]=dict_categoies_result
        dict_categoies_result={}

    return dict_word_by_topic_result,dict_topic_length

def findTopicwithDirectory(dir,trainedData,topicData):
    word_array=[]
    result_by_category={}
    result_dict={}
    result_array=[]
    for root, subdirs, files in os.walk(dir):
        for file in files:
            with open(root+"/"+file, mode='rt', encoding='utf-8') as f:
                content = f.readlines()
                word_array = textClassify(content)
                f.close()
            for category in categories:
                wordList=trainedData[category]
                p_array=[]
                for w in word_array:
                    if w in wordList:
                        p = trainedData[category][w]
                    else:
                        p = 0.1
                    p_array.append(p)
                
                r=0
                for p in p_array:
                    r += math.log(p)
                
                r += math.log(topicData[category])

                result_by_category[category] = r
    
            result = max(result_by_category, key=result_by_category.get)
            
            result_dict[result]=result_by_category
            result_array.append(result_dict)
    return result_array       

def findTopicwithFile(dir,trainedData,topicData):
    word_array=[]
    result_by_category={}
    with open(dir, mode='rt', encoding='utf-8') as f:
        content = f.readlines()
        word_array = textClassify(content)
        f.close()
    for category in categories:
        wordList=trainedData[category]
        p_array=[]
        sum = 0
        for w in word_array:
            if w in wordList:
                p = trainedData[category][w]
            else:
                p = 0.00001
            p_array.append(p)
            sum += p
        r=0
        
        for p in p_array:
            r += math.log(p)
        
        r += math.log(topicData[category])
     
        result_by_category[category] = r
    print(result_by_category)
    result = max(result_by_category, key=result_by_category.get)
    return result        

#########################################################
#get Training File From Directory and save Probability words for test 
#########################################################
for category in categories:
    cateogory_dict[category]=[]
    for root, subdirs, files in os.walk('dataset/dev/'+category):
        for file in files:
            with open(root+"/"+file, mode='rt', encoding='utf-8') as f:
                content=f.readlines()
                getTextFromFile(content,category)
                f.close()

trainedData,topicData= training(cateogory_dict)

########################################################
#traing data result
########################################################
# print(findTopicwithDirectory('dataset/dev/interest',trainedData,topicData))
# print(findTopicwithDirectory('dataset/dev/jobs',trainedData,topicData))
# print(findTopicwithDirectory('dataset/dev/money_supply',trainedData,topicData))
# print(findTopicwithDirectory('dataset/dev/trade',trainedData,topicData))

########################################################
#test data result
########################################################
print(findTopicwithFile('dataset/test/1.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/2.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/3.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/4.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/5.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/6.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/7.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/8.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/9.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/10.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/11.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/12.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/13.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/14.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/15.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/16.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/17.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/18.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/19.txt',trainedData,topicData))
print(findTopicwithFile('dataset/test/20.txt',trainedData,topicData))