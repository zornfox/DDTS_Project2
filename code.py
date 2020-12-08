import os
import numpy as np
from decimal import Decimal
import re
import csv
import string
from stemming.porter2 import stem
import itertools
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
import random
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn import svm
import scipy
from scipy import sparse
import collections
import math

##################################################**** Part1 ****#######################################################
"""
Part 1: IR EVALUATION ---> Method of reading required files
"""


def read_file_q1(data_path, encoding='utf-8'):
    result_list = []
    with open(data_path, 'r', encoding=encoding) as f:
        for line in f.readlines():
            result_list.append(re.sub(',', ' ', line.strip()).split())
    return result_list


"""
Part 1: IR EVALUATION ----> Read files of Part 1
"""
system_results_list = read_file_q1('system_results.csv')
qrels_list = read_file_q1('qrels.csv')

"""
Part 1: IR EVALUATION ----> Method of extracting relevant doc list for each query
"""


def extract_rel_docID():
    relevant_docs_list = []
    for i in range(1, 11):
        relevant_docs_for_eachQuery = [sublist[1] for sublist in qrels_list if sublist[0] == str(i)]
        relevant_docs_list.append(relevant_docs_for_eachQuery)
    return relevant_docs_list


# RelevDocs_list = extract_rel_docID()
# print(RelevDocs_list[0])

"""
Part 1: IR EVALUATION ----> Method of extracting relevant doc list and relevance for each query
"""


def extract_Relevance():
    Relevance_list = []
    for i in range(1, 11):
        relevant_docs_for_eachQuery = [sublist[2] for sublist in qrels_list if sublist[0] == str(i)]
        Relevance_list.append(relevant_docs_for_eachQuery)
    return Relevance_list


"""
Part 1: IR EVALUATION ----> insert mean value to the list 
                            and convert all element to 3 decimal points and to string
"""


def Calculate_mean(original_list):
    # add mean value for each system,Given 10 queries in total
    for i in range(6):
        mean = sum(original_list[i * 10 + i * 1: 10 + i * 10 + i * 1]) / 10
        original_list.insert(10 + i * 10 + i * 1, mean)
    # convert each element to string with 3 decimal points
    correct_format_list = [str(Decimal(elem).quantize(Decimal('0.000'))) for elem in original_list]
    return correct_format_list


"""
Part 1: IR EVALUATION ----> P@10: precision at cutoff N 
                            and return a result of P@10 with mean value
"""


def P_AT_N(real_results_list, N):
    Precision_list = []
    RelevDocs_list = extract_rel_docID()
    for i in range(1, 7):
        for j in range(1, 11):
            RelevDocs_for_query = RelevDocs_list[j - 1]
            realDocs = [sublist[2] for sublist in real_results_list if
                        sublist[0] == str(i) and sublist[1] == str(j)]
            realDocs_at_N = realDocs[:N]
            TP = list(set(RelevDocs_for_query).intersection(set(realDocs_at_N)))
            P = len(TP) / N
            Precision_list.append(P)
    Add_mean_list = Precision_list.copy()
    precision_str_list = Calculate_mean(Add_mean_list)
    return precision_str_list


"""
Part 1: IR EVALUATION ----> R@50: recall at cutoff 50.
                            and return a result of R@50 with mean value 
"""


def R_AT_N(real_results_list, N):
    Recall_list = []
    RelevDocs_list = extract_rel_docID()
    for i in range(1, 7):
        for j in range(1, 11):
            RelevDocs_for_query = RelevDocs_list[j - 1]
            realDocs = [sublist[2] for sublist in real_results_list if
                        sublist[0] == str(i) and sublist[1] == str(j)]
            realDocs_at_N = realDocs[:N]
            TP = list(set(RelevDocs_for_query).intersection(set(realDocs_at_N)))
            P = len(TP) / len(RelevDocs_for_query)
            Recall_list.append(P)
    Add_mean_list = Recall_list.copy()
    recall_str_list = Calculate_mean(Add_mean_list)
    return recall_str_list


"""
Part 1: IR EVALUATION ----> r_Precision
                            and return a result of r_Precision with mean value 
"""


def RP_AT_rank(real_results_list):
    RP_list = []
    RelevDocs_list = extract_rel_docID()
    for i in range(1, 7):
        for j in range(1, 11):
            RelevDocs_for_query = RelevDocs_list[j - 1]
            rank = len(RelevDocs_for_query)
            realDocs = [sublist[2] for sublist in real_results_list if
                        sublist[0] == str(i) and sublist[1] == str(j)]
            realDocs_at_N = realDocs[:rank]
            TP = list(set(RelevDocs_for_query).intersection(set(realDocs_at_N)))
            P = len(TP) / rank
            RP_list.append(P)
    Add_mean_list = RP_list.copy()
    recall_str_list = Calculate_mean(Add_mean_list)
    return recall_str_list


"""
Part 1: IR EVALUATION ----> Average Precision
                            and return a result of Average Precision with mean value 
"""


def AP(real_results_list):
    AP_list = []
    RelevDocs_list = extract_rel_docID()
    for i in range(1, 7):
        for j in range(1, 11):
            RelevDocs_for_query = RelevDocs_list[j - 1]
            realDocs = [sublist[2] for sublist in real_results_list if
                        sublist[0] == str(i) and sublist[1] == str(j)]
            intersec = list(set(RelevDocs_for_query).intersection(set(realDocs)))
            position = 0
            retr_num = 0
            retr_Pre = []
            for each in realDocs:
                position += 1
                if each in RelevDocs_for_query:
                    retr_num += 1
                    Precision_each = retr_num / float(position)
                    retr_Pre.append(Precision_each)
            if not len(intersec) == 0:
                AP_list.append(sum(retr_Pre) / len(RelevDocs_for_query))
            else:
                AP_list.append(0)
    Add_mean_list = AP_list.copy()
    AP_str_list = Calculate_mean(Add_mean_list)
    return AP_str_list


"""
Part 1: IR EVALUATION ----> Method of calculating DCG at N and return a result of DCG_at_N_list
"""


def DCG_at_N(real_results_list, N):
    DCG_list = []
    RelevDocs_list = extract_rel_docID()
    Relevance_list = extract_Relevance()
    for i in range(1, 7):
        for j in range(1, 11):
            RelevDocs_for_query = RelevDocs_list[j - 1]
            Relevance_for_query = Relevance_list[j - 1]
            realDocs = [sublist[2] for sublist in real_results_list if
                        sublist[0] == str(i) and sublist[1] == str(j)]
            realDocs_at_N = realDocs[:N]
            TP = list(set(RelevDocs_for_query).intersection(set(realDocs_at_N)))
            dcg = 0
            for indx, ele in enumerate(realDocs_at_N):
                if indx == 0:
                    if ele in TP:
                        pointer = RelevDocs_for_query.index(ele)
                        dcg += int(Relevance_for_query[pointer])
                    else:
                        dcg += 0
                else:
                    if ele in TP:
                        pointer = RelevDocs_for_query.index(ele)
                        rank_temp = indx + 1
                        dcg += int(Relevance_for_query[pointer]) / np.log2(rank_temp)
                    else:
                        dcg += 0
            DCG_list.append(dcg)
    return DCG_list


"""
Part 1: IR EVALUATION ----> Method of calculating iDCG at N and return a result of iDCG_at_N_list 
"""


def iDCG_at_N(N):
    iDCG_list = []
    RelevDocs_list = extract_rel_docID()
    Relevance_list = extract_Relevance()
    for i in range(1, 7):
        for j in range(1, 11):
            RelevDocs_for_query = RelevDocs_list[j - 1]
            Relevance_for_query = Relevance_list[j - 1]
            RelevDocs_at_N = RelevDocs_for_query[:N]
            Relevance_at_N = Relevance_for_query[:N]
            idcg = 0
            for indx, ele in enumerate(RelevDocs_at_N):
                if indx == 0:
                    idcg += int(Relevance_at_N[0])
                else:
                    idcg += int(Relevance_at_N[indx]) / np.log2(indx + 1)
            iDCG_list.append(idcg)
    return iDCG_list


"""
Part 1: IR EVALUATION ----> Method of calculating nDCG at N 
                            and return a result of nDCG_at_N_list with mean value 
"""


def nDCG_at_N(real_results_list, N):
    DCG_N_List = DCG_at_N(real_results_list, N)
    iDCG_N_List = iDCG_at_N(N)
    nDCG_N_List = []
    for indx, ele in enumerate(DCG_N_List):
        nDCG_N_List.append(ele / iDCG_N_List[indx])
    Add_mean_list = nDCG_N_List.copy()
    nDCG_str_list = Calculate_mean(Add_mean_list)
    return nDCG_str_list


"""
Part 1: IR EVALUATION ----> Method of writing to the cvs file
"""


def execute(data_path, encoding='utf-8'):
    P_at_10 = P_AT_N(system_results_list, 10)
    R_at_50 = R_AT_N(system_results_list, 50)
    r_P = RP_AT_rank(system_results_list)
    AP_list = AP(system_results_list)
    nDCG_10 = nDCG_at_N(system_results_list, 10)
    nDCG_20 = nDCG_at_N(system_results_list, 20)

    table_data = []

    s_index = 1
    q_index = 1
    for i in range(66):
        row = []
        s = s_index
        if (i + 1) % 11 == 0:
            s_index += 1
            q_index = 'mean'
        q1 = q_index
        if q_index == 'mean':
            q_index = 1
        else:
            q_index += 1
        row.append(s)
        row.append(q1)
        row.append(P_at_10[i])
        row.append(R_at_50[i])
        row.append(r_P[i])
        row.append(AP_list[i])
        row.append(nDCG_10[i])
        row.append(nDCG_20[i])
        table_data.append(tuple(row))
    print(table_data)
    with open(data_path, 'w', encoding=encoding, newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['system_number', 'query_number', 'P@10', 'R@50', 'r-precision', 'AP', 'nDCG@10', 'nDCG@20'])
        writer.writerows(table_data)


"""
Part 1: IR EVALUATION ----> execute
"""
if not os.path.exists("ir_eval.csv"):
    execute('ir_eval.csv')

##################################################**** Part2 ****#######################################################
"""
Part 2: Text Analysis ----> Method of Reading files
                          and do half preprocessing including remove all the punctuations and make each element to lower
"""


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

quran_D, quran_T = DocCollection_and_termCollection('quran')
ot_D, ot_T = DocCollection_and_termCollection('ot')
nt_D, nt_T = DocCollection_and_termCollection('nt')

"""
Part 2: Text Analysis ----> Calculate Mutual information and Chi-square for each unique term for each corpora
                            where input is target class, non-target class 1 and 2
"""


def MI_and_ChiSquare(target_class, term_collection, non_target_1, non_target_2):
    Non_target_list = non_target_1 + non_target_2
    N = len(target_class) + len(Non_target_list)
    Collection_MI = []
    Collection_Chi_Square = []
    for et in term_collection:
        word_MI = []
        word_Chi_square = []
        N11, N10 = 0, 0
        P1, P2, P3, P4 = 0, 0, 0, 0
        for doc in target_class:
            if et in doc:
                N11 += 1
        N01 = len(target_class) - N11
        for dc in Non_target_list:
            if et in dc:
                N10 += 1
        N00 = len(Non_target_list) - N10
        if N * N11 / ((N11 + N10) * (N11 + N01)) != 0:
            P1 = np.log2(N * N11 / ((N11 + N10) * (N11 + N01)))
        if N * N01 / ((N01 + N00) * (N11 + N01)) != 0:
            P2 = np.log2(N * N01 / ((N01 + N00) * (N11 + N01)))
        if N * N10 / ((N11 + N10) * (N10 + N00)) != 0:
            P3 = np.log2(N * N10 / ((N11 + N10) * (N10 + N00)))
        if N * N00 / ((N01 + N00) * (N10 + N00)) != 0:
            P4 = np.log2(N * N00 / ((N01 + N00) * (N10 + N00)))
        MI = (N11 / N) * P1 + (N01 / N) * P2 + (N10 / N) * P3 + (N00 / N) * P4
        Chi_square = (N * ((N11 * N00 - N10 * N01) ** 2)) / ((N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00))
        word_MI.append(et)
        word_MI.append(MI)
        word_Chi_square.append(et)
        word_Chi_square.append(Chi_square)
        Collection_MI.append(word_MI)
        Collection_Chi_Square.append(word_Chi_square)
    sorted_collection_MI = sorted(Collection_MI, key=lambda x: x[1], reverse=True)
    sorted_collection_Chi = sorted(Collection_Chi_Square, key=lambda x: x[1], reverse=True)
    return sorted_collection_MI[0:10], sorted_collection_Chi[0:10]


"""
Part 2: Text Analysis ----> Method of writing MI and Chi-square results to txt
"""


def write_results(data_path, encoding='utf-8'):
    quran_MI, quran_Chi = MI_and_ChiSquare(quran_D, quran_T, ot_D, nt_D)
    ot_MI, ot_Chi = MI_and_ChiSquare(ot_D, ot_T, quran_D, nt_D)
    nt_MI, nt_Chi = MI_and_ChiSquare(nt_D, nt_T, quran_D, ot_D)
    with open(data_path, 'w', encoding=encoding) as f1:
        f1.write("For Quran: top 10 MI results\n")
        for each in quran_MI:
            f1.write(str(each) + "\n")
        f1.write("For Quran: top 10 Chi_square results\n")
        for each in quran_Chi:
            f1.write(str(each) + "\n")
        f1.write("For OT: top 10 MI results\n")
        for each in ot_MI:
            f1.write(str(each) + "\n")
        f1.write("For OT: top 10 Chi_square results\n")
        for each in ot_Chi:
            f1.write(str(each) + "\n")
        f1.write("For NT: top 10 MI results\n")
        for each in nt_MI:
            f1.write(str(each) + "\n")
        f1.write("For NT: top 10 Chi_square results\n")
        for each in nt_Chi:
            f1.write(str(each) + "\n")
    return quran_MI, quran_Chi, ot_MI, ot_Chi, nt_MI, nt_Chi


"""
Part 2: Text Analysis ----> execute writing method 
"""
if not os.path.exists('MI_&_ChiSquare.txt'):
    write_results('MI_&_ChiSquare.txt')

"""
Part 2: Text Analysis ----> LDA
"""
common_texts = quran_D + ot_D + nt_D
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
lda = LdaModel(common_corpus, num_topics=20, random_state=1000, id2word=common_dictionary)

"""
Part 2: Text Analysis ----> method of finding top 3 average score for each topic
                            and top 10 tokens for the top 3
"""


def Top3_and_Top10(corpora_doc_list):
    all_list = []
    for q in corpora_doc_list:
        each = common_dictionary.doc2bow(q)
        all_list.append(lda.get_document_topics(bow=each, minimum_probability=0.00))
    flatten = itertools.chain.from_iterable
    score_collection = list(flatten(all_list))
    average_list = []
    for i in range(20):
        prob_list = [item[1] for item in score_collection if item[0] == i]
        summing = sum(prob_list)
        aver = summing / len(corpora_doc_list)
        average_list.append([i, aver])
    average_sorted = sorted(average_list, key=lambda x: x[1], reverse=True)
    Top3_Topic = average_sorted[0:3]
    Top10_token = []
    for ec in Top3_Topic:
        Top10_token.append(lda.show_topic(ec[0]))
    return Top3_Topic, Top10_token


"""
Part 2: Text Analysis ----> method of saving results to txt file
"""


def LDA_save(data_path, encoding='utf-8'):
    quran_topic_top3, quran_token_top10 = Top3_and_Top10(quran_D)
    ot_topic_top3, ot_top10 = Top3_and_Top10(ot_D)
    nt_topic_top3, nt_top10 = Top3_and_Top10(nt_D)
    with open(data_path, 'w', encoding=encoding) as f2:
        f2.write("For Quran: top 3 average score with following top 10 tokens for this topic\n")
        for i, item in enumerate(quran_topic_top3):
            f2.write(str(item) + "\n")
            f2.write(str(quran_token_top10[i]) + "\n")
        f2.write("For OT: top 3 average score with following top 10 tokens for this topic\n")
        for i, item in enumerate(ot_topic_top3):
            f2.write(str(item) + "\n")
            f2.write(str(ot_top10[i]) + "\n")
        f2.write("For NT: top 3 average score with following top 10 tokens for this topic\n")
        for i, item in enumerate(nt_topic_top3):
            f2.write(str(item) + "\n")
            f2.write(str(nt_top10[i]) + "\n")
    return quran_topic_top3, quran_token_top10, ot_topic_top3, ot_top10, nt_topic_top3, nt_top10


"""
Part 2: Text Analysis ----> write LDA results 
"""
if not os.path.exists('LDA_results.txt'):
    LDA_save('LDA_results.txt')

############################################**** Part3-Baseline ****###################################################
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
Trn_set, Tst_set = train_test_split(Dataset, train_size=0.2, test_size=0.8, random_state=0)
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
word2id = {}
for word_id, word in enumerate(Trn_vocab):
    word2id[word] = word_id
cls2id = {}
for cls_id, cls in enumerate(set(Trn_class)):
    cls2id[cls] = cls_id
# print("word id with word")
# print(word2id)
print("class id with class")
print(cls2id)

"""
convert data to BOW format 
"""


def convert_to_bow_matrix(doc_Data, word2id):
    matrix_size = (len(doc_Data), len(word2id) + 1)
    # out of words index
    oov_index = len(word2id)
    # matrix index by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)

    # iterate through all docs in the dataset
    for dc_id, dc in enumerate(doc_Data):
        for wrd in dc:
            X[dc_id, word2id.get(wrd, oov_index)] += 1
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
    num_correct = 0
    num_total = len(pred)
    error_index = []
    for i, (p, t) in enumerate(zip(pred, true)):
        if p == t:
            num_correct += 1
        if p != t:
            error_index.append([i, p, t])
    for each in error_index:
        each.append(original_doc_collection[each[0]])
    return num_correct / num_total, error_index


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
acc_tst, error_tst = compute_accuracy(Y_tst_prediction, Y_tst, Tst_doc)
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
acc_new, error_new = compute_accuracy(Y_New_prediction, Y_New, New_doc)
print("accuracy on development set")
print(acc_new)
print("error for tst")
print(error_new)

"""
calculate P, R, F1
"""


def computing(pred, true):
    TP, FP, FN = 0, 0, 0
    label_list = [cls2id['quran'], cls2id['ot'], cls2id['nt']]
    compu_res = []
    for lb in label_list:
        for p, t in zip(pred, true):
            if t == lb and p == lb:
                TP += 1
            if t != lb and p == lb:
                FP += 1
            if t == lb and p != lb:
                FN += 1
        lb_P = TP / (TP + FP)
        compu_res.append(lb_P)
        lb_R = TP / (TP + FN)
        compu_res.append(lb_R)
        lb_F1 = 2 * lb_P * lb_R / (lb_P + lb_R)
        compu_res.append(lb_F1)
    Pmacro = sum(compu_res[0:3]) / 3
    compu_res.append(Pmacro)
    Rmacro = sum(compu_res[3:6]) / 3
    compu_res.append(Rmacro)
    Fmacro = sum(compu_res[6:9]) / 3
    compu_res.append(Fmacro)
    return compu_res




############################################**** Part3-Improved ****###################################################
"""
select training set and testing set 
"""
Dataset_imp = All_corpora_list.copy()
random.shuffle(Dataset_imp)
Trn_set_imp, Tst_set_imp = train_test_split(Dataset_imp, train_size=0.9, test_size=0.1, random_state=0)

Trn_doc_imp, Trn_class_imp, Trn_vocab_imp = AllDocList_ClassList_VocabList(Trn_set_imp)
Tst_doc_imp, Tst_class_imp, Tst_vocab_imp = AllDocList_ClassList_VocabList(Tst_set_imp)

"""
set up mappings for word and class IDS
"""
word2id_imp = {}
for word_id, word in enumerate(Trn_vocab_imp):
    word2id_imp[word] = word_id
cls2id_imp = {}
for cls_id, cls in enumerate(set(Trn_class_imp)):
    cls2id_imp[cls] = cls_id

"""
term frequency and doc frequency 
"""
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

"""
convert data to BOW format 
"""
def convert_to_bow_matrix_imp(doc_Data, word2id):
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
Train SVM model 
"""
X_trn_imp = convert_to_bow_matrix_imp(Trn_doc_imp, word2id=word2id_imp)
Y_trn_imp = [cls2id_imp[cat] for cat in Trn_class_imp]
model_imp = sklearn.svm.SVC(C=1000)
model_imp.fit(X_trn_imp, Y_trn_imp)

"""
predict train dev test
"""
Y_train_prediction_imp = model_imp.predict(X_trn_imp)
X_tst_imp = convert_to_bow_matrix_imp(Tst_doc_imp, word2id=word2id_imp)
Y_tst_imp = [cls2id_imp[cat] for cat in Tst_class_imp]
Y_tst_prediction_imp = model_imp.predict(X_tst_imp)
X_New_imp = convert_to_bow_matrix_imp(New_doc, word2id=word2id_imp)
Y_New_imp = [cls2id_imp[cat] for cat in New_class]
Y_New_prediction_imp = model_imp.predict(X_New_imp)

"""
print result
"""
# print("\n\n")
# print(computing(Y_train_prediction_imp, Y_trn_imp))
# print(computing(Y_tst_prediction_imp,Y_tst_imp))
# print(computing(Y_New_prediction_imp,Y_New_imp))

def writePart3(data_path, encoding='utf-8'):
    row1=computing(Y_train_prediction, Y_trn)
    row1.insert(0, 'train')
    row1.insert(0, 'baseline')
    row2=computing(Y_tst_prediction, Y_tst)
    row2.insert(0, 'dev')
    row2.insert(0, 'baseline')
    row3=computing(Y_New_prediction, Y_New)
    row3.insert(0, 'test')
    row3.insert(0, 'baseline')
    row4=computing(Y_train_prediction_imp, Y_trn_imp)
    row4.insert(0, 'train')
    row4.insert(0, 'improved')
    row5=computing(Y_tst_prediction_imp,Y_tst_imp)
    row5.insert(0, 'dev')
    row5.insert(0, 'improved')
    row6=computing(Y_New_prediction_imp, Y_New_imp)
    row6.insert(0, 'test')
    row6.insert(0, 'improved')
    with open(data_path, 'w', encoding=encoding, newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['system', 'split', 'p-quran', 'r-quran', 'f-quran', 'p-ot', 'r-ot', 'f-ot','p-nt',
                         'r-nt', 'f-nt','p-macro','r-macro','f-macro'])
        writer.writerow(row1)
        writer.writerow(row2)
        writer.writerow(row3)
        writer.writerow(row4)
        writer.writerow(row5)
        writer.writerow(row6)

if not os.path.exists('classification.csv'):
    writePart3('classification.csv')