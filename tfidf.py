
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy
from LoadDocs import get_data, conllu_parse
from CleanData import compile_doc_data


def tfidf(documents, reduced_documents=None):
    """Tokenise inputted documents and return raw tf*idf vectors of inputted documents"""
    if reduced_documents:
        vocab = sorted(list(set([
            v.lower() for v in [
                # [x for y in z for x in y] formula combines the contents of all sub-lists within a list:
                # It takes a list, z, of sub-lists, y, containing items, x (in this case tokens)
                # A new list is created containing each x from each y for each y in z
                x for y in [i.split(" ") for i in reduced_documents] for x in y]
        ])))
    else:
        vocab = None
    vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r"(?u)\b[\wáéíóúↄḟṁṅæǽ⁊ɫ֊̃]+\b", vocabulary=vocab)
    vectors = vectorizer.fit_transform(documents)
    return vectors


def k_medoids(documents, classifier, reduced_documents=None):
    """Returns raw k_medoid vectors for raw tf*idf vectors from inputted documents"""
    tfidf_doc = tfidf(documents, reduced_documents)
    kmedoids = classifier.fit_predict(tfidf_doc)
    return kmedoids


def pca_2d(documents, pca_classifier, reduced_documents=None):
    """Performs Principal Component Analysis on documents inputted, returns an n-dimensional array of PCA dense vectors
       (usually a 2D array, number of dimensions depends on pca_classifier)"""
    tfidf_doc = tfidf(documents, reduced_documents)
    PCA_2D = pca_classifier.fit_transform(tfidf_doc.todense())
    return PCA_2D


def pca_centres(classifier, pca_classifier):
    """Returns PCA dense vectors for center points of each cluster"""
    centres = pca_classifier.transform(classifier.cluster_centers_.todense())
    return centres


def draw_subplots(data, colors, plotname, clusters, centres=None, cmap='viridis', header='Old Irish Gloss Clusters'):

    plot = plotname

    plot.axhline(0, color='#afafaf')
    plot.axvline(0, color='#afafaf')

    for i in range(clusters):
        try:
            plot.scatter(data[i:, 0], data[i:, 1], s=30, c=colors, cmap=cmap)
        except (KeyError, ValueError) as e:
            pass

    if isinstance(centres, numpy.ndarray):
        plot.scatter(centres[:, 0], centres[:, 1], marker="x", c='r')

    plot.set_xlabel('Principal Component 1')
    plot.set_ylabel('Principal Component 2')

    plot.set_title(header)


if __name__ == "__main__":

    # # Select Wb. Data

    # # Tokenisation style 1

    # # All Word-types, Natural Tokens, No Features
    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1))

    # # Function-words Only, Natural Tokens, No Features
    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1), True)

    # # All Word-types, Tokens Standardised, No Features
    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1), False, True)

    # # Function-words Only, Tokens Standardised, No Features
    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1), True, True)

    # # Function-words Only, Tokens Standardised, Features Added
    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1), True, True, True)

    # # Tokenisation style 2

    # # All Word-types, Natural Tokens, No Features
    wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2))

    # # Function-words Only, Natural Tokens, No Features
    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2), True)

    # # All Word-types, Tokens Standardised, No Features
    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2), False, True)

    # # Function-words Only, Tokens Standardised, No Features
    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2), True, True)

    # # Function-words Only, Tokens Standardised, Features Added
    # wb_data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2), True, True, True)

    # # Select Sg. Data

    # # Tokenisation style 1

    # # All Word-types, Natural Tokens, No Features
    # sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"))

    # # Function-words Only, Natural Tokens, No Features
    # sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"), True)

    # # All Word-types, Tokens Standardised, No Features
    # sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"), False, True)

    # # Function-words Only, Tokens Standardised, No Features
    # sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"), True, True)

    # # Function-words Only, Tokens Standardised, Features Added
    # sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"), True, True, True)

    # # Tokenisation style 2

    # # All Word-types, Natural Tokens, No Features
    sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"))

    # # Function-words Only, Natural Tokens, No Features
    # sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"), True)

    # # All Word-types, Tokens Standardised, No Features
    # sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"), False, True)

    # # Function-words Only, Tokens Standardised, No Features
    # sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"), True, True)

    # # Function-words Only, Tokens Standardised, Features Added
    # sg_data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"), True, True, True)


    # Test Wb.
    docs = wb_data[0]
    hand_names = wb_data[-1]
    reduced_docs = None
    if len(wb_data) == 3:
        reduced_docs = wb_data[1]
    clusters = 4

    # # Test Sg.
    # docs = sg_data[0]
    # hand_names = sg_data[-1]
    # reduced_docs = None
    # if len(sg_data) == 3:
    #     reduced_docs = sg_data[1]
    # clusters = 3

    hl_dict = {}
    handcount = 0
    for hand_name in sorted(list(set(hand_names))):
        hl_dict[hand_name] = handcount
        handcount += 1
    hand_labels = [hl_dict.get(i) for i in hand_names]

    classifier = KMedoids(n_clusters=clusters, metric="cosine", random_state=0)
    km = k_medoids(docs, classifier, reduced_docs)
    pca_classifier = PCA(n_components=2)
    pca_2d_matrix = pca_2d(docs, pca_classifier, reduced_docs)
    centres_matrix = pca_centres(classifier, pca_classifier)

    fig, (plot1, plot2) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20, 10))
    draw_subplots(pca_2d_matrix, classifier.labels_, plot1, clusters, centres_matrix, header='Colours = Clusters')
    draw_subplots(pca_2d_matrix, hand_labels, plot2, clusters, header='Colours = Topics (Hands)')
    plt.show()

    # # TEST FUNCTIONS

    # # Provide Test Data
    #
    # doc1 = "These are some words I am putting in a document."
    # doc2 = "This document is comprised of a number of words."
    # doc3 = "Some of the words in this document are found in other documents also."
    # doc4 = "We are now writing a piece of text which is entirely separate from the others and, hence, dissimilar."
    # doc5 = "Another piece of writing in which we are interested for its dissimilarity to its precursors is this."
    # docs = [doc1, doc2, doc3, doc4, doc5]
    #
    # reduced_doc1 = "These I in a"
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
