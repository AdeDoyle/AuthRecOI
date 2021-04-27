
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_extra.cluster import KMedoids
import numpy as np
import matplotlib.pyplot as plt


def tfidf(documents):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names, index=["doc1", "doc2", "doc3"])
    return df


def k_medoids(tfidf_doc):
    tfidf_array = np.asarray(tfidf_doc)
    kmedoids = KMedoids(n_clusters=3, metric='cosine', random_state=0).fit(tfidf_array)
    return kmedoids


if __name__ == "__main__":

    doc1 = "These are some words I'm putting in a document."
    doc2 = "This document is comprised of a number of words."
    doc3 = "Some of the words in this document are found in other documents also."

    docs = [doc1, doc2, doc3]

    # # TEST FUNCTIONS

    # # Test tfidf function

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(tfidf(docs))

    # # Test k_medoids function

    km = k_medoids(tfidf(docs))
    # print(km.labels_)
    # print(km.cluster_centers_)
    # print(km.medoid_indices_)
    # print(km.inertia_)

