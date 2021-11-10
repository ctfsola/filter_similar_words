import jieba                                           
import re 
#from numba import njit, prange
#jieba.load_userdict("newdict.txt")                     #加载自定义词典  
import jieba.posseg as pseg 
import difflib                # 方法一：
from fuzzywuzzy import fuzz   # 方法二：
from collections import Counter
import numpy as np
import os
#把停用词做成字典
stopwords = {}
fstop = open(os.path.dirname(__file__)+'/stop_words.txt', 'r',encoding='utf-8',errors='ignore')
for eachWord in fstop:
    stopwords[eachWord.strip().encode('utf-8').decode('utf-8', 'ignore')] = eachWord.strip().encode('utf-8').decode('utf-8', 'ignore')
fstop.close()

# 方法三：编辑距离，又称Levenshtein距离
#@njit(parallel = True)
def edit_similar(str1, str2):   # str1，str2是分词后的标签列表
    len_str1 = len(str1)
    len_str2 = len(str2)
    taglist = np.zeros((len_str1+1, len_str2+1))
    for a in range(len_str1):
        taglist[a][0] = a
    for a in range(len_str2):
        taglist[0][a] = a
    for i in range(1, len_str1+1):
        for j in range(1, len_str2+1):
            if(str1[i - 1] == str2[j - 1]):
                temp = 0
            else:
                temp = 1
            taglist[i][j] = min(taglist[i - 1][j - 1] + temp, taglist[i][j - 1] + 1, taglist[i - 1][j] + 1)
    return 1-taglist[len_str1][len_str2] / max(len_str1, len_str2)

# 方法四：余弦相似度
def cos_sim(str1, str2):        # str1，str2是分词后的标签列表
    co_str1 = (Counter(str1))
    co_str2 = (Counter(str2))
    p_str1 = []
    p_str2 = []
    for temp in set(str1 + str2):
        p_str1.append(co_str1[temp])
        p_str2.append(co_str2[temp])
    p_str1 = np.array(p_str1)
    p_str2 = np.array(p_str2)
    return p_str1.dot(p_str2) / (np.sqrt(p_str1.dot(p_str1)) * np.sqrt(p_str2.dot(p_str2)))

# def trip_word(row):
#     line = row.strip().encode('utf-8').decode('utf-8', 'ignore')
#     return re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "",line)


def cut_stop_word(wordList):
    outStr = ''
    for word in wordList:
        if word not in stopwords:  
            outStr += word
    return outStr
    #return ''.join([word.strip() for word in wordList if word not in stopwords])

def check_similar(str1,str2):
    #jieba.enable_parallel(4)    #并行分词
    diff_result1 = difflib.SequenceMatcher(None, str1, str2).ratio() #方法一，好像是完形匹配算法
    diff_result2 = fuzz.ratio(str1, str2)/100   # 方法2

    str11 = cut_stop_word(jieba.cut(str1))
    str22 = cut_stop_word(jieba.cut(str2))
    diff_result3 = edit_similar(str11,str22)
    diff_result4 = cos_sim(str11,str22)
    # print(diff_result1,diff_result2,diff_result3,diff_result4)
    # print(sum((diff_result1,diff_result2,diff_result3,diff_result4))/4)
    return sum((diff_result1,diff_result2,diff_result3,diff_result4))/4

    # print(diff_result1,diff_result2)
    # return sum((diff_result1,diff_result2))/2

def splitSentence(inputFile, outputFile,threshold):
    #fin = open(inputFile, 'r',encoding='utf-8')        #以读的方式打开文件  
    with open(inputFile,encoding='utf-8') as f:
        fin = [x.strip() for x in f.readlines()]
    fout = open(outputFile, 'w',encoding='utf-8')

    for i,row in enumerate(fin):
        for left_word in fin[i+1:]:
            if check_similar(row, left_word) > threshold:
                fout.write(row + '\n')
                break
    fout.close()  

threshold = input('请输入阈值：')
if not threshold:
    threshold = 0.5

path = os.path.dirname(__file__)+'/'
splitSentence(path+'input.txt', path+'different.txt',float(threshold) )


