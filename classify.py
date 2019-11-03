import numpy as np
import os
import csv
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


def read_label(data_dir):
    f = open(os.path.join(data_dir, 'label.txt'))
    docs = f.readlines()
    y_true = np.array([int(doc.strip())-1 for doc in docs])

    return y_true


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


def calc_rep(docs, word_emb):
    emb = [np.array([word_emb[w] for w in doc if w in word_emb]) for doc in docs]
    emb = np.array([np.average(vec, axis=0) for vec in emb])
    return emb


def f1(y_true, y_pred):
    assert y_pred.size == y_true.size
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    return f1_macro, f1_micro


def classify(train, train_label, test, test_label, k=3):
    print(f"Using KNN, k = {k}")
    neigh = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=20)
    neigh.fit(train, train_label)
    y_pred = neigh.predict(test)
    f1_macro, f1_micro = f1(test_label, y_pred)
    print(f"F1 macro: {f1_macro}, F1 micro: {f1_micro}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='classify',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='20news')
    parser.add_argument('--emb_file', default='jose.txt')
    parser.add_argument('--train_num', default=-1, type=int)

    args = parser.parse_args()
    print(args)

    print(f'### Test: Document Classification ###')
    doc_emb = get_emb(vec_file=os.path.join("datasets", args.dataset, args.emb_file))
    y_true = read_label(os.path.join("datasets", args.dataset))
    
    train_label = y_true[:args.train_num]
    test_label = y_true[args.train_num:]
    train = doc_emb[:args.train_num]
    test = doc_emb[args.train_num:]

    classify(train, train_label, test, test_label, k=3)
