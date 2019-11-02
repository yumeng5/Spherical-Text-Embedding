import numpy as np
import os
import argparse
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics


def get_emb(vec_file):
    f = open(vec_file, 'r')
    tmp = f.readlines()
    contents = tmp[1:]
    doc_emb = np.zeros([int(x) for x in tmp[0].split(' ')])
    
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        doc_emb[i] = np.array(vec)

    return doc_emb


def read_label(data_dir):
    f = open(os.path.join(data_dir, 'label.txt'))
    docs = f.readlines()
    y_true = np.array([int(doc.strip())-1 for doc in docs])
    return y_true


def cluster_doc(doc_emb, K, method):
    y_pred = []
    if method == "kmeans":
        # k-means
        print("Clustering using K-Means")
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=1)
        km.fit(doc_emb)
        y_pred = km.labels_
    elif method == "skmeans":
        # spherical k-means
        print("Clustering using Spherical K-Means")
        from spherecluster import SphericalKMeans
        skm = SphericalKMeans(n_clusters=K, n_init=1)
        skm.fit(doc_emb)
        y_pred = skm.labels_
    return y_pred


def purity_score(y_true, y_pred):
    # compute confusion matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def cal_metric(y_pred, y_true):
    s1 = mutual_info_score(y_pred, y_true)
    print(f'mutual_info_score = {s1}')
    s2 = normalized_mutual_info_score(y_pred, y_true)
    print(f'normalized_mutual_info_score = {s2}')
    s3 = adjusted_rand_score(y_pred, y_true)
    print(f'adjusted_rand_score = {s3}')
    s4 = purity_score(y_true, y_pred)
    print(f'purity = {s4}')
    return [s1, s2, s3, s4]


def write_res(file_dir, res):
    f = open(file_dir, 'a')
    f.write(','.join([str(r) for r in res]) + '\n')
    return 


def calc_rep(docs, word_emb):
    emb = [np.array([word_emb[w] for w in doc if w in word_emb]) for doc in docs]
    emb = np.array([np.average(vec, axis=0) for vec in emb])
    return emb


def get_avg_emb(vec_file, text):
    f = open(vec_file, 'r')
    contents = f.readlines()[1:]
    word_emb = {}
    vocabulary = {}
    vocabulary_inv = {}
    for i, content in enumerate(contents):
        content = content.strip()
        tokens = content.split(' ')
        word = tokens[0]
        vec = tokens[1:]
        vec = [float(ele) for ele in vec]
        word_emb[word] = np.array(vec)
        vocabulary[word] = i
        vocabulary_inv[i] = word

    f = open(text, 'r')
    contents = f.readlines()
    doc_emb = np.zeros((len(contents), len(word_emb[vocabulary_inv[0]])))
    for i, content in enumerate(contents):
        content = content.strip()
        doc = content.split(" ")
        emb = np.array([word_emb[w] for w in doc if w in word_emb])
        doc_emb[i] = np.average(emb, axis=0)

    return doc_emb


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='20news')
    parser.add_argument('--emb_file', default='jose.txt')
    parser.add_argument('--method', choices=['kmeans','skmeans'])
    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--corpus', default='text.txt')

    args = parser.parse_args()
    print(args)

    print(f'### Test: Document Clustering ###')
    doc_emb = get_emb(vec_file=os.path.join("./datasets", args.dataset, args.emb_file))
    y_pred = cluster_doc(doc_emb, args.k, args.method)
    y_true = read_label(os.path.join("./datasets", args.dataset))
    res = cal_metric(y_pred, y_true)

    