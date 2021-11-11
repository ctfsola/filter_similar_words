from numba import njit, prange
import jieba                                           
import re 
#jieba.load_userdict("newdict.txt")                     #加载自定义词典  
import jieba.posseg as pseg 
import difflib                # 方法一：
from fuzzywuzzy import fuzz   # 方法二：
from collections import Counter
import numpy as np

# import jieba 
# import numpy as np
# from collections import Counter
#print(np.array([1,2,3]))

# 方法三：编辑距离，又称Levenshtein距离
# @njit
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


def check_similar(str1,str2):
    str11 = jieba.lcut(str1)
    str22 = jieba.lcut(str2)
    diff_result3 = edit_similar(str11,str22)
    diff_result4 = cos_sim(str11,str22)
    print(diff_result3,diff_result4)
    return sum((diff_result3,diff_result4))/2

res = check_similar('谁是喜洋洋','喜洋洋与灰太狼')
print(res)