from gensim.models import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
import re
import string
from stemming.porter2 import stem
import itertools
#######################
import random
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import svm
import scipy
from scipy import sparse
import collections
import math
#######################


def read_file_q2(data_path, encoding='utf-8'):
    HalfPreprocess_list = []
    with open(data_path, 'r', encoding=encoding) as f:
        for line in f.readlines():
            # remove all the punctuations and make each element to lower
            single = re.sub(r'[{}]+'.format(string.punctuation), ' ', line.strip().lower()).split()
            HalfPreprocess_list.append(single)
    return HalfPreprocess_list


"""
Part 2: Text Analysis ----> Method of removing stop words and stemming, input is 'englishST.txt'
"""


def prepreocessing(data_path, encoding='utf-8'):
    stopwords = []
    finished_pre = []
    half_preprocess = read_file_q2('train_and_dev.tsv')
    with open(data_path, encoding=encoding) as f:
        for stop in f:
            stopwords.append(stop.strip('\n'))
    f.close()
    for ele in half_preprocess:
        single = [stem(i) for i in ele if i not in stopwords]
        finished_pre.append(single)
    return finished_pre


"""
Part 2: Text Analysis ----> Method of obtaining Doc collection and unique term collection for each corpora
"""


def DocCollection_and_termCollection(corpora_name):
    All_corpora_list = prepreocessing('englishST.txt')
    doc_collection = [doc[1:] for doc in All_corpora_list if doc[0] == corpora_name]
    # flatten the list
    temp = doc_collection.copy()
    flatten = itertools.chain.from_iterable
    words_collection = list(flatten(temp))
    unique_term_collection = list(set(words_collection))
    return doc_collection, unique_term_collection


"""
Part 2: Text Analysis ----> get Doc collection and unique word collection for each corpora
"""

# quran_D, quran_T = DocCollection_and_termCollection('quran')
# print(len(quran_D))
# ot_D, ot_T = DocCollection_and_termCollection('ot')
# print(len(ot_D))
# nt_D, nt_T = DocCollection_and_termCollection('nt')
# print(len(nt_D))



########################################################### Part3 ####################################################
"""
get data collection
"""
All_corpora_list = prepreocessing('englishST.txt')
print(len(All_corpora_list))
"""
select training set and testing set 
"""
Dataset = All_corpora_list.copy()
random.shuffle(Dataset)
Trn_set, Tst_set = train_test_split(Dataset, train_size=0.9, test_size=0.1,random_state=0)
print(len(Trn_set))
print(len(Tst_set))

"""
Convert to list of lists: documents containing tokens 
and return categories 
also get the Vocabulary
"""
def AllDocList_ClassList_VocabList(data):
    doc_collection = [doc[1:] for doc in data]
    Classes_collection = [doc[0] for doc in data]
    # flatten the list
    temp = doc_collection.copy()
    flatten = itertools.chain.from_iterable
    words_collection = list(flatten(temp))
    Vocab_collection = set(words_collection)
    return doc_collection, Classes_collection, Vocab_collection

"""
get three list for both training and testing 
"""
Trn_doc, Trn_class, Trn_vocab = AllDocList_ClassList_VocabList(Trn_set)
Tst_doc, Tst_class, Tst_vocab = AllDocList_ClassList_VocabList(Tst_set)
print("For training doc size", len(Trn_doc))
print(Trn_doc[0:10])
print("For training class size", len(Trn_class))
print(Trn_class[0:10])
print("For training vocab size", (len(Trn_vocab)))
print("For tst doc size", len(Tst_doc))
print(Trn_doc[0:10])
print("For tst class size", len(Tst_class))
print(Trn_class[0:10])
print("For tst vocab size", len(Tst_vocab))


"""
check the most common category in training set 
"""
print("\n")
print("check the most common category in training set ")
print(collections.Counter(Trn_class).most_common())

"""
set up mappings for word and class IDS
"""
word2id ={}
for word_id, word in enumerate(Trn_vocab):
    word2id[word]=word_id
cls2id = {}
for cls_id, cls in enumerate(set(Trn_class)):
    cls2id[cls]=cls_id
# print("word id with word")
# print(word2id)
print("class id with class")
print(cls2id)
########################################## For improving
##########################################
def InvPosDic(doc_dta):
    term_docNo_Posi=[]
    for i,doc in enumerate(doc_dta):
        for j,term in enumerate(doc):
            term_docNo_Posi.append((term,i,j))

    Sort_term_docNo_Posi=sorted(term_docNo_Posi)
    inverted_positional_index = {}
    for each in Sort_term_docNo_Posi:
        word_str = each[0]
        docNo_int = each[1]
        pos_int = each[2]
        # check if there is a key existing
        if word_str not in inverted_positional_index.keys():
            inverted_positional_index.update({word_str: {docNo_int: [pos_int]}})
        else:
            # if the key exits , checking the document number
            if docNo_int not in inverted_positional_index[word_str].keys():
                # inverted_positional_index.update(inverted_positional_index[word_str].update({docNo_int: [pos_int]}))
                inverted_positional_index[word_str][docNo_int] = [pos_int]
            else:
                inverted_positional_index[word_str][docNo_int].append(pos_int)
    return inverted_positional_index


def weight(tf,df,doc_Data):
    return (1 + math.log(tf, 10)) * math.log(len(doc_Data)/df, 10)
####################################
####################################
"""
convert data to BOW format 
"""
def convert_to_bow_matrix(doc_Data, word2id):
    inv_posi_index_Dic=InvPosDic(doc_Data)
    matrix_size = (len(doc_Data), len(word2id)+1)
    # out of words index
    oov_index= len(word2id)
    # matrix index by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)

    #iterate through all docs in the dataset
    for dc_id, dc in enumerate(doc_Data):
        for wrd in dc:
            df=len(inv_posi_index_Dic[wrd])
            tf=len(inv_posi_index_Dic[wrd][dc_id])
            X[dc_id,word2id.get(wrd, oov_index)] += weight(tf,df,doc_Data)
    return X


"""
doc collection and labels to be predicted
"""
X_trn = convert_to_bow_matrix(Trn_doc, word2id=word2id)
Y_trn = [cls2id[cat] for cat in Trn_class]
print(X_trn[:3])
print(Y_trn[:3])

"""
Train SVM model 
"""
model = sklearn.svm.SVC(C=1000)
model.fit(X_trn, Y_trn)


"""
Evaluate training data and return the mis_predicting doc
"""
Y_train_prediction = model.predict(X_trn)
def compute_accuracy(pred, true, original_doc_collection):
    num_correct=0
    num_total=len(pred)
    error_index=[]
    for i,(p,t) in enumerate(zip(pred, true)):
        if p==t:
            num_correct+=1
        if p!=t:
            error_index.append([i,p,t])
    for each in error_index:
        each.append(original_doc_collection[each[0]])
    return num_correct/num_total, error_index


acc_train, error_trn = compute_accuracy(Y_train_prediction, Y_trn, Trn_doc)
print("accuracy on training set")
print(acc_train)
print("error for train")
print(error_trn)

"""
for development set 
"""
X_tst = convert_to_bow_matrix(Tst_doc, word2id=word2id)
Y_tst = [cls2id[cat] for cat in Tst_class]
Y_tst_prediction = model.predict(X_tst)
acc_tst, error_tst = compute_accuracy(Y_tst_prediction,Y_tst,Tst_doc)
print("accuracy on development set")
print(acc_tst)
print("error for tst")
print(error_tst)

"""
pre_process new test data
"""
def prepreocessing2(data_path, encoding='utf-8'):
    stopwords = []
    finished_pre = []
    half_preprocess = read_file_q2('test.tsv')
    with open(data_path, encoding=encoding) as f:
        for stop in f:
            stopwords.append(stop.strip('\n'))
    f.close()
    for ele in half_preprocess:
        single = [stem(i) for i in ele if i not in stopwords]
        finished_pre.append(single)
    return finished_pre

New_test_list = prepreocessing2('englishST.txt')
print("new data length", len(New_test_list))
New_doc, New_class, New_vocab = AllDocList_ClassList_VocabList(New_test_list)
print("new doc length", len(New_doc))
print("new class length", len(New_class))
print("new vocab length", len(New_vocab))
X_New = convert_to_bow_matrix(New_doc, word2id=word2id)
Y_New = [cls2id[cat] for cat in New_class]
Y_New_prediction = model.predict(X_New)
acc_new, error_new = compute_accuracy(Y_New_prediction,Y_New,New_doc)
print("accuracy on development set")
print(acc_new)
print("error for tst")
print(error_new)

"""
calculate P, R, F1
"""
def computing(pred, true):
    TP, FP, FN = 0,0,0
    label_list=[cls2id['quran'],cls2id['ot'],cls2id['nt']]
    compu_res = []
    for lb in label_list:
        for p,t in zip(pred, true):
            if t==lb and p==lb:
                TP += 1
            if t!=lb and p==lb:
                FP += 1
            if t ==lb and p!=lb:
                FN += 1
        lb_P = TP/(TP+FP)
        compu_res.append(lb_P)
        lb_R = TP/(TP+FN)
        compu_res.append(lb_R)
        lb_F1 = 2*lb_P*lb_R/(lb_P+lb_R)
        compu_res.append(lb_F1)
    Pmacro=sum(compu_res[0:3])/3
    compu_res.append(Pmacro)
    Rmacro = sum(compu_res[3:6]) / 3
    compu_res.append(Rmacro)
    Fmacro=sum(compu_res[6:9])/3
    compu_res.append(Fmacro)
    return compu_res

print("\n\n")
print(computing(Y_train_prediction, Y_trn))
print(computing(Y_tst_prediction,Y_tst))
print(computing(Y_New_prediction,Y_New))


