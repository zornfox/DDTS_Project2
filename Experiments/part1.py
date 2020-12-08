import os
import numpy as np
from decimal import Decimal
import re
import csv
import string
from stemming.porter2 import stem
import itertools

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
            position=0
            retr_num=0
            retr_Pre=[]
            for each in realDocs:
                position+=1
                if each in RelevDocs_for_query:
                    retr_num += 1
                    Precision_each=retr_num/float(position)
                    retr_Pre.append(Precision_each)
            if not len(intersec) == 0:
                AP_list.append(sum(retr_Pre)/len(RelevDocs_for_query))
            else:
                AP_list.append(0)
    Add_mean_list = AP_list.copy()
    AP_str_list = Calculate_mean(Add_mean_list)
    return AP_str_list


"""
Part 1: IR EVALUATION ----> Method of calculating DCG at N and return a result of DCG_at_N_list
"""
def DCG_at_N(real_results_list, N):
    DCG_list=[]
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
            dcg=0
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
                        rank_temp = indx+1
                        dcg += int(Relevance_for_query[pointer]) / np.log2(rank_temp)
                    else:
                        dcg += 0
            DCG_list.append(dcg)
    return DCG_list

"""
Part 1: IR EVALUATION ----> Method of calculating iDCG at N and return a result of iDCG_at_N_list 
"""
def iDCG_at_N(N):
    iDCG_list=[]
    RelevDocs_list = extract_rel_docID()
    Relevance_list = extract_Relevance()
    for i in range(1, 7):
        for j in range(1, 11):
            RelevDocs_for_query = RelevDocs_list[j - 1]
            Relevance_for_query = Relevance_list[j - 1]
            RelevDocs_at_N = RelevDocs_for_query[:N]
            Relevance_at_N = Relevance_for_query[:N]
            idcg=0
            for indx, ele in enumerate(RelevDocs_at_N):
                if indx == 0:
                    idcg += int(Relevance_at_N[0])
                else:
                    idcg += int(Relevance_at_N[indx]) / np.log2(indx+1)
            iDCG_list.append(idcg)
    return iDCG_list


"""
Part 1: IR EVALUATION ----> Method of calculating nDCG at N 
                            and return a result of nDCG_at_N_list with mean value 
"""
def nDCG_at_N(real_results_list, N):
    DCG_N_List=DCG_at_N(real_results_list, N)
    iDCG_N_List=iDCG_at_N(N)
    nDCG_N_List=[]
    for indx, ele in enumerate(DCG_N_List):
        nDCG_N_List.append(ele/iDCG_N_List[indx])
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



