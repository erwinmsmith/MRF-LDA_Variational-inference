from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
from collections import Counter
from scipy import spatial
from textblob import TextBlob
import re
import jieba

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.special import gammaln
import scipy

# 将输入列表 k 中的数字转换成单词
def convert_numbers(k):
    for i in range(len(k)):
        try:
            num2words(int(k[i]))
            k[i] = " "
        except:
            pass
    return k

# 计算余弦相似度
def get_cosine(a, b):
    return 1 - spatial.distance.cosine(a, b)

# 将所有文本转换为小写，然后移除特殊字符和标点符号
def preprocess(pd):
    pd = pd.str.lower()
    pd = pd.str.replace('[{}]'.format('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\\n\\t'), ' ')
    pd = pd.apply(lambda x: [w for w in word_tokenize.tokenize(x)])
    pd = pd.apply(lambda x: convert_numbers(x))
    pd = pd.str.join(' ')
    pd = pd.apply(lambda x: [PorterStemmer().lemmatize(w) for w in word_tokenize.tokenize(x)])
    pd = pd.apply(lambda x: [item for item in x if item not in stopwords.words('english')])
    return pd

# 输出词频矩阵、TF-IDF矩阵、词汇表和词汇表的逆映射
def processReviews(reviews, window=5, MAX_VOCAB_SIZE=1000):
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None)
    count_matrix = vectorizer.fit_transform(reviews)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
    words = vectorizer.get_feature_names_out()
    vocabulary = dict(zip(words, np.arange(len(words))))
    inv_vocabulary = dict(zip(np.arange(len(words)), words))
    return count_matrix.toarray(), tfidf_matrix.toarray(), vocabulary, words

# 定义jieba分词和去停用词的函数
def jieba_cut(mytext):
    with open('baidu_stopwords.txt', encoding='utf-8') as f:
        file_stop = f.readlines()
        stop_list = [re.sub(r'\s+', '', line).strip() for line in file_stop]
    seg_list = jieba.lcut(mytext)
    word_list = [word for word in seg_list if word not in stop_list]
    return " ".join(word_list)

# KL散度计算
from scipy.stats import entropy

def kl_score(pk, qk):
    return (entropy(pk, qk) * 0.5 + entropy(qk, pk) * 0.5)

# 一致性得分计算
def coherence_score(X, topic_word_dist, vocabulary):
    X[X > 1] = 1
    totalcnt = len(topic_word_dist)
    total = 0
    for topic_idx, words in topic_word_dist.items():
        for word1 in words:
            for word2 in words:
                if word1 != word2:
                    ind1 = vocabulary.get(word1)
                    ind2 = vocabulary.get(word2)
                    if ind1 is not None and ind2 is not None:
                        total += np.log((np.matmul(X[:, ind1].T, X[:, ind2]) + 1.0) / np.sum(X[:, ind2]))
    return total / (2 * totalcnt)

# H分数计算
def get_hscore(doc_topic_dist, X, k):
    testlen = X.shape[0]
    all_kl_scores = np.zeros((testlen, testlen))
    for i in range(testlen - 1):
        for j in range(i + 1, testlen):
            score = kl_score(doc_topic_dist[i], doc_topic_dist[j])
            all_kl_scores[i, j] = score
            all_kl_scores[j, i] = score

    dt = np.zeros((X.shape[0], k))

    for i in range(X.shape[0]):
        dt[i, np.argmax(doc_topic_dist[i])] = 1

    intradist = 0
    for i in range(k):
        cnt = dt[:, i].sum()
        tmp = np.outer(dt[:, i], dt[:, i])
        tmp = tmp * all_kl_scores
        intradist += tmp.sum() * 1.0 / (cnt * (cnt - 1))
    intradist = intradist / k

    interdist = 0
    for i in range(k):
        for j in range(k):
            if i != j:
                cnt_i = dt[:, i].sum()
                cnt_j = dt[:, j].sum()
                tmp = np.outer(dt[:, i], dt[:, j])
                tmp = tmp * all_kl_scores
                interdist += tmp.sum() * 1.0 / (cnt_i * cnt_j)
    interdist = interdist / (k * (k - 1))

    return intradist / interdist