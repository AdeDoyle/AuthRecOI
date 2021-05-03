
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from LoadDocs import get_data, conllu_parse
from CleanData import compile_hand_data, compile_doc_data


# the logarithm of the number of documents in the corpus divided by


def tfidf(documents, reduced_documents=None):
    """Tokenise inputted documents and return raw tf*idf vectors of inputted documents"""
    vectorizer = TfidfVectorizer(analyzer='word')
    vectors = vectorizer.fit_transform(documents)

    # vectors = []
    #
    # if not reduced_documents:
    #     reduced_documents = documents
    #
    # N = len(documents)
    #
    # for docnum, d in enumerate(documents):
    #     rd = reduced_documents[docnum]
    #     maxrf = max([d.count(i) for i in d])
    #     for t in rd:
    #         tfd = rd.count(t)
    #         tf = tfd/maxrf
    #         df = 1  # Start count at 1 for smoothing
    #         for doc in reduced_documents:
    #             if t in doc:
    #                 df += 1
    #         idf = math.log(N/df)
    #         vectors.append(tf*idf)

    # for i in vectors:
    #     print(i)
    print(vectors)

    return vectors


def k_medoids(documents, classifier):
    """Returns raw k_medoid vectors for raw tf*idf vectors from inputted documents"""
    tfidf_doc = tfidf(documents)
    kmedoids = classifier.fit_predict(tfidf_doc)
    return kmedoids


def pca_2d(documents, pca_classifier):
    """Performs Principal Component Analysis on documents inputted, returns an n-dimensional array of PCA dense vectors
       (usually a 2D array, number of dimensions depends on pca_classifier)"""
    tfidf_doc = tfidf(documents)
    PCA_2D = pca_classifier.fit_transform(tfidf_doc.todense())
    return PCA_2D


def pca_centres(classifier, pca_classifier):
    """Returns PCA dense vectors for center points of each cluster"""
    centres = pca_classifier.transform(classifier.cluster_centers_.todense())
    return centres


def draw_subplots(data, colors, plotname, clusters, centres='empty', cmap='viridis', header='Old Irish Gloss Clusters'):

    plot = plotname

    plot.axhline(0, color='#afafaf')
    plot.axvline(0, color='#afafaf')

    for i in range(clusters):
        try:
            plot.scatter(data[i:, 0], data[i:, 1], s=30, c=colors, cmap=cmap)
        except (KeyError, ValueError) as e:
            pass

    if centres != 'empty':
        plot.scatter(centres[:, 0], centres[:, 1], marker="x", c='r')

    plot.set_xlabel('Principal Component 1')
    plot.set_ylabel('Principal Component 2')

    plot.set_title(header)


if __name__ == "__main__":

    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json")))
    wb_data = compile_hand_data(conllu_parse(get_data("Wb. Manual Tokenisation.json")))
    docs = wb_data[0]
    hand_names = wb_data[1]
    hl_dict = {}
    handcount = 0
    for hand_name in sorted(list(set(hand_names))):
        hl_dict[hand_name] = handcount
        handcount += 1
    hand_labels = [hl_dict.get(i) for i in hand_names]

    clusters = 3

    classifier = KMedoids(n_clusters=clusters, metric="cosine", random_state=0)
    km = k_medoids(docs, classifier)
    pca_classifier = PCA(n_components=2)
    pca_2d_matrix = pca_2d(docs, pca_classifier)
    centres_matrix = pca_centres(classifier, pca_classifier)

    fig, (plot1, plot2) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20, 10))
    draw_subplots(pca_2d_matrix, classifier.labels_, plot1, clusters, centres_matrix, header='colours = clusters')
    draw_subplots(pca_2d_matrix, hand_labels, plot2, clusters, header='colours = topics')
    plt.show()

    # # TEST FUNCTIONS

    # # Provide Test Data
    #
    # doc1 = "These are some words I'm putting in a document."
    # doc2 = "This document is comprised of a number of words."
    # doc3 = "Some of the words in this document are found in other documents also."
    # doc4 = "We are now writing a piece of text which is entirely separate from the others and, hence, dissimilar."
    # doc5 = "Another piece of writing in which we are interested for its dissimilarity to its precursors is this."
    # docs = [doc1, doc2, doc3, doc4, doc5]
    #
    # reduced_doc1 = "These I'm in a"
    # reduced_doc2 = "This of a of"
    # reduced_doc3 = "Some of the in this in"
    # reduced_doc4 = "We a of which from the and"
    # reduced_doc5 = "Another of in which we for its to its this"
    # reduced_docs = [reduced_doc1, reduced_doc2, reduced_doc3, reduced_doc4, reduced_doc5]
    #
    # hand_labels = [0, 1, 1, 2, 2]
    # hand_names = ["hand 3", "hand 1", "hand 1", "hand 2", "hand 2"]


    # # Test tfidf function
    #
    # tfidf_doc = tfidf(docs, reduced_docs)
    #
    # print(tfidf_doc)

    # # # Test k_medoids function
    #
    # classifier = KMedoids(n_clusters=clusters, metric="cosine", random_state=0)
    # km = k_medoids(docs, classifier)
    #
    # print(classifier.labels_)
    # print(classifier.cluster_centers_)
    # print(classifier.medoid_indices_)
    # print(classifier.inertia_)

    # # # Test pca, pca_2d and pca_centres functions
    #
    # pca_classifier = pca(2)
    # pca_2d_matrix = pca_2d(docs, pca_classifier)
    # centres_matrix = pca_centres(classifier, pca_classifier)

    # Test draw_subplots function

    # fig, (plot1, plot2) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20, 10))
    # draw_subplots(pca_2d_matrix, classifier.labels_, plot1, clusters, header='colours = clusters')
    # draw_subplots(pca_2d_matrix, hand_labels, plot2, clusters, header='colours = topics')
    # plt.show()
