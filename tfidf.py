
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import numpy as np
from math import log
from sklearn.preprocessing import normalize as norm
from scipy import sparse
from LoadDocs import get_data, conllu_parse
from CleanData import compile_doc_data


def tf(docs):
    """Returns a term-frequency dictionary containing every term which occurs in a collection of documents.
       The key is the term as a string.
       The value is a list of integers, where each integer represents the number of times a particular term is used in
       a given document, and the list contains an integer for each document in the collection of documents.
       Format: {"term1"; [int-doc1, int-doc2, int-doc3, ...], "term2"; [int-doc1, int-doc2, int-doc3, ...], ...}"""
    termfreqs = dict()
    doc = " ".join(docs)
    doc = doc.split(" ")
    doc = sorted(list(set(doc)))
    for term in doc:
        termfreqs[term] = [sing_doc.count(term) for sing_doc in docs]
    return termfreqs


def idf(term, docs):
    """Returns the inverse document frequency for a single term in a collection of documents."""
    big_d = len(docs)
    denominator = 0
    for doc in docs:
        if term in doc:
            denominator += 1
    return log(big_d/(1 + denominator))


def idf_doc(terms, docs):
    """Returns a dictionary of inverse document frequencies for a list of terms in a collection of documents.
       The key is the term as a string.
       The value is the inverse document frequency for that term in the collection of documents (probably a float).
       Format: {"term1"; float-doc1, "term2"; float-doc2, ...}"""
    termidfs = dict()
    for term in terms:
        termidfs[term] = idf(term, docs)
    return termidfs


def tfidf_manual(docs):
    """Returns raw tf*idf vectors of inputted documents.
       Vectors generated manually."""
    matrix = list()
    # Get Term Frequencies dict for each unique term in all docs
    termfreqs = tf(docs)
    # Get Inverse document frequencies dictionary for each term in all docs
    termidfs = idf_doc(termfreqs, docs)
    # Multiply each term frequency by its inverse document frequency
    for term in termfreqs:
        termfreqs[term] = [i * termidfs.get(term) for i in termfreqs.get(term)]
        # Add an array of tf*idf scores (one for each document) for this term to the matrix
        matrix.append(termfreqs.get(term))
    # Reorientate the matrix (flip it left to right, then rotate 90 degrees counterclockwise)
    # Now the vectors for each document are top-to-bottom instead of left-to-right
    matrix = np.rot90(np.fliplr(np.array(matrix)))
    # # Find L2 Norm (Euclidean distance) of vectors in matrix
    matrix = norm(matrix, axis=1, norm='l2')
    # Turn the matrix into a compressed sparse row matrix
    matrix = sparse.csr_matrix(matrix)
    return matrix


def tfidf(documents, reduced_documents=None, manual=False):
    """Tokenise inputted documents and return raw tf*idf vectors of inputted documents
       Vectors generated using sklearn's TfidfVectorizer."""
    if manual:
        return tfidf_manual(documents)
    else:
        # Create a vocab list containing only unique tokens from the reduced documents
        if reduced_documents:
            vocab = sorted(list(set([
                v.lower() for v in [
                    # [x for y in z for x in y] formula combines the contents of all sub-lists within a list:
                    # It takes a list, z (all reduced documents), of sub-lists, y (individual reduced documents),
                    # each containing items, x (in this case tokens).
                    # From these a new list is created containing each x from each y for each y in z
                    x for y in [i.split(" ") for i in reduced_documents] for x in y]
            ])))
        else:
            vocab = None
        # Vectorise the content of the documents (or reduced documents) using TfidfVectorizer
        # token pattern makes it possible to tokenise based on the unicode contents of Irish datasets
        # (unicode flag + word boundary + 1 or more possible characters for Old Irish tokens + word boundary)
        vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r"(?u)\b[\wáéíóúↄḟṁṅæǽ⁊ɫ֊̃]+\b", vocabulary=vocab,
                                     smooth_idf=True)
        vectors = vectorizer.fit_transform(documents)
        return vectors


def build_classifier(classifier_type, num_clusters, metric="cosine", method="pam", random_state=0):
    """Returns k-means or k_medoid classifier model"""
    if classifier_type == "k-means":
        classifier_model = KMeans(num_clusters)
    elif classifier_type == "k-medoids":
        classifier_model = KMedoids(num_clusters, metric=metric, method=method, random_state=random_state)
    else:
        raise RuntimeError(f"No classifier, '{classifier_type}', is possible, only 'k-means' or 'k-medoids'.")
    return classifier_model


def reduce_dimensions(tfidf_vecs, reduction_type, classifier_type, n_components=2,
                      learning_rate='auto', init='random', perplexity=30):
    """Performs Principal Component Analysis on documents inputted, returns an n-dimensional array of PCA dense vectors
       (usually a 2D array, number of dimensions depends on pca_classifier)"""

    if reduction_type == "PCA":
        # Create PCA classifier
        pca_classifier = PCA(n_components)
        # Fit PCA Classifier to tf*idf data
        PCA_2D = pca_classifier.fit_transform(tfidf_vecs.todense())
        # Get PCA dense vectors for center points of each cluster
        try:
            centres_matrix = pca_classifier.transform(classifier_type.cluster_centers_)
        except TypeError:
            centres_matrix = pca_classifier.transform(classifier_type.cluster_centers_.todense())
        return [PCA_2D, centres_matrix]
    elif reduction_type == "t-SNE":
        # Create t-SNE classifier
        tsne_classifier = TSNE(n_components=n_components, learning_rate=learning_rate, init=init, perplexity=perplexity)
        # Fit t-SNE Classifier to tf*idf data
        tsne_matrix = tsne_classifier.fit_transform(tfidf_vecs.todense())
        return [tsne_matrix]
    else:
        raise RuntimeError(f"No dimensionality reduction method ,'{reduction_type}', is possible, "
                           f"only 'PCA' or 't-SNE'.")


def draw_subplots(subplot, subplot_title, matrix_data, colour_numbers, colour_labels=None,
                  shape_numbers=None, shape_labels=None, centres=None):
    """Draws an individual sub-plot of cluster data to be fit within an overall plot"""

    # Initially identify the plot variable
    plot = subplot

    # Add horizontal and vertical lines to the plot across the axis
    plot.axhline(0, color='#afafaf')
    plot.axvline(0, color='#afafaf')

    # Ensure colour and shape numbers are lists, not numpy arrays, so that they can be indexed
    if isinstance(colour_numbers, np.ndarray):
        colour_numbers = list(colour_numbers)
    if isinstance(shape_numbers, np.ndarray):
        shape_numbers = list(shape_numbers)

    # Get a sorted list of the individual numbers corresponding to data intended to be represented using colour
    unique_colour_numbers = sorted(list(set(colour_numbers)))

    # If labels have been provided corresponding to the colour data, create a corresponding list
    # containing each unique label, sorted to match the sorting of the colour data numbers list
    unique_colour_labels = None
    if colour_labels:
        # If a single string has been passed instead of a list of colour labels
        # Generate a list of colour labels by combining the string with each of the colour numbers (+1 for readability)
        if isinstance(colour_labels, str):
            colour_labels = [f"{colour_labels} {sorted_num + 1}" for sorted_num in colour_numbers]
        unique_colour_labels = list()
        for unique_colour_num in unique_colour_numbers:
            first_inst = colour_numbers.index(unique_colour_num)
            unique_colour_labels.append(colour_labels[first_inst])
    # If no colour labels have been provided, create a list of corresponding labels numbering the data (starting at 1)
    else:
        unique_colour_labels = [f"Grouping {sorted_num + 1}" for sorted_num in shape_numbers]

    # Identify a discrete colour palette to utilise
    col_dict = mplcol.TABLEAU_COLORS
    col_list = [i for i in col_dict]

    # Identify a dictionary of acceptable shapes and appropriate sizes to utilise
    shape_dict = {"o": 30, "p": 50, "*": 60, "d": 30, "X": 60, "s": 30, "^": 30, "H": 60, "8": 60}
    shape_list = [i for i in shape_dict]

    # If shape data has been passed as an argument as well as colour data
    unique_shape_numbers = None
    if isinstance(shape_numbers, list):

        # Get a sorted list of the individual numbers corresponding to data intended to be represented using shapes
        unique_shape_numbers = sorted(list(set(shape_numbers)))

        # If labels have been provided corresponding to the shape data, create a corresponding list
        # containing each unique label, sorted to match the sorting of the shape data numbers list
        unique_shape_labels = None
        if shape_labels:
            # If a single string has been passed instead of a list of shape labels
            # Generate a list of shape labels by combining the string with each of the shape numbers (+1 as above)
            if isinstance(shape_labels, str):
                shape_labels = [f"{shape_labels} {sorted_num + 1}" for sorted_num in shape_numbers]
            unique_shape_labels = list()
            for unique_shape_num in unique_shape_numbers:
                first_inst = shape_numbers.index(unique_shape_num)
                unique_shape_labels.append(shape_labels[first_inst])
        # If no shape labels have been provided, create a list of labels numbering the data (starting at 1)
        else:
            unique_shape_labels = [f"Cluster {sorted_num + 1}" for sorted_num in shape_numbers]

    # Raise an error if shape data has been passed as anything other than a list or an array
    elif shape_numbers:
        raise RuntimeError("Shape number data not in list form")

    # If shape data has been passed as an argument as well as colour data
    if isinstance(shape_numbers, list):
        # For each unique grouping (generated cluster or identified hand) in the colour data
        for unique_colour_num in unique_colour_numbers:
            # For each unique grouping (generated cluster or identified hand) in the shape data
            for unique_shape_num in unique_shape_numbers:
                colour_shape_data = list()
                # For every data point (matrix coordinates) in the matrix data
                for data_index, coordinates in enumerate(matrix_data):
                    # If that data point is linked to the unique grouping (hand and cluster combined)
                    # i.e. both the colour number at the same index in the colour numbers list and
                    # the shape number at the same index in the shape numbers list match the unique grouping
                    # Add that data point to the colour/shape data list
                    if (colour_numbers[data_index] == unique_colour_num
                            and shape_numbers[data_index] == unique_shape_num):
                        colour_shape_data.append(coordinates)
                colour_shape_data = np.array(colour_shape_data)

                # Once the colour data list of coordinates for that unique grouping is complete plot it on the graph
                if len(colour_shape_data) > 0:
                    x = [i[0] for i in colour_shape_data]
                    y = [i[1] for i in colour_shape_data]
                    shape = shape_list[unique_shape_num]
                    col_label = unique_colour_labels[unique_colour_num]
                    if unique_shape_labels:
                        shape_label = unique_shape_labels[unique_shape_num]
                        combined_label = f"{col_label} / {shape_label}"
                    else:
                        combined_label = col_label
                    plot.scatter(x, y, s=shape_dict.get(shape), c=col_dict.get(col_list[unique_colour_num - 1]),
                                 marker=shape, label=combined_label)

    # If only colour data has been passed as an argument
    elif not shape_numbers:
        # For each unique grouping (generated cluster or identified hand) in the colour data
        for unique_colour_num in unique_colour_numbers:
            colour_data = list()
            # For every data point (matrix coordinates) in the matrix data
            for i, coordinates in enumerate(matrix_data):
                # If that data point is linked to the unique grouping (hand or cluster)
                # i.e. the colour number at the same index in the colour numbers list matches the unique grouping
                # Add that data point to the colour data list
                if colour_numbers[i] == unique_colour_num:
                    colour_data.append(coordinates)
            colour_data = np.array(colour_data)

            # Once the colour data list of coordinates for that unique grouping is complete plot it on the graph
            x = [i[0] for i in colour_data]
            y = [i[1] for i in colour_data]
            plot.scatter(x, y, s=30, c=col_dict.get(col_list[unique_colour_num - 1]),
                         label=unique_colour_labels[unique_colour_num])

    # Testing to see if centres data is not null in the usual way will cause a ValueError if it is an array
    # Because centres data can either be null or an array, test to see if it is an array
    # If so, mark centres data on the scatter plot
    if isinstance(centres, np.ndarray):
        plot.scatter(centres[:, 0], centres[:, 1], marker="x", s=100, c='r')

    # Include the legend in the plot in the best available position (avoiding any datapoints)
    plot.legend(loc="best", fontsize="small")

    # Include the subplot title in the plot
    plot.set_title(subplot_title)


if __name__ == "__main__":

    # # Select Wb. Data

    # # Tokenisation style 1

    # # All Word-types, Natural Tokens, No Features
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1))

    # # Function-words Only, Natural Tokens, No Features
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1), True)

    # # All Word-types, Tokens Standardised, No Features
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1), False, True)

    # # Function-words Only, Tokens Standardised, No Features
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1), True, True)

    # # All Word-types, Tokens Standardised, Features Added
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1), False, True, True)

    # # Function-words Only, Tokens Standardised, Features Added
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=1), True, True, True)

    # # Tokenisation style 2

    # # All Word-types, Natural Tokens, No Features
    data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2))

    # # Function-words Only, Natural Tokens, No Features
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2), True)

    # # All Word-types, Tokens Standardised, No Features
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2), False, True)

    # # Function-words Only, Tokens Standardised, No Features
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2), True, True)

    # # All Word-types, Tokens Standardised, Features Added
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2), False, True, True)

    # # Function-words Only, Tokens Standardised, Features Added
    # data = compile_doc_data(conllu_parse(get_data("Wb. Manual Tokenisation.json"), tok_style=2), True, True, True)


    # # Select Sg. Data

    # # Tokenisation style 1

    # # All Word-types, Natural Tokens, No Features
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"))

    # # Function-words Only, Natural Tokens, No Features
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"), True)

    # # All Word-types, Tokens Standardised, No Features
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"), False, True)

    # # Function-words Only, Tokens Standardised, No Features
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"), True, True)

    # # All Word-types, Tokens Standardised, Features Added
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"), False, True, True)

    # # Function-words Only, Tokens Standardised, Features Added
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_combined_POS.conllu"), True, True, True)

    # # Tokenisation style 2

    # # All Word-types, Natural Tokens, No Features
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"))

    # # Function-words Only, Natural Tokens, No Features
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"), True)

    # # All Word-types, Tokens Standardised, No Features
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"), False, True)

    # # Function-words Only, Tokens Standardised, No Features
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"), True, True)

    # # All Word-types, Tokens Standardised, Features Added
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"), False, True, True)

    # # Function-words Only, Tokens Standardised, Features Added
    # data = compile_doc_data(get_data("sga_dipsgg-ud-test_split_POS.conllu"), True, True, True)

    reduced_docs = None
    if len(data) == 3:
        reduced_docs = data[1]
    docs = data[0]
    hand_names = data[-1]
    hand_label_dict = {}
    handcount = 0
    for hand_name in sorted(list(set(hand_names))):
        hand_label_dict[hand_name] = handcount
        handcount += 1
    hand_numbers = [hand_label_dict.get(i) for i in hand_names]


    # # Set Parameters

    # # Select Clustering Algorithm
    # classification = "k-means"
    classification = "k-medoids"

    # # Set number of Clusters
    clusters = 3

    # # Select Dimensionality Reduction Method
    d_reduce = "PCA"
    # d_reduce = "t-SNE"

    # # Set Dimensionality Reduction Parameters
    n_components = 2  # Dimension of the embedded space.
    # learning_rate = 'auto'
    learning_rate = 10.0  # The learning rate for t-SNE is usually in the range [10.0, 1000.0]
    init = 'random'
    # init = 'pca'  # PCA initialization is usually more globally stable than random initialization.
    perplexity = 30  # The perplexity must be less than the number of samples.


    # # Plot tf*idf of Data

    # # Get tf*idf of documents
    tfidf_vectors = tfidf(docs, reduced_docs)

    # # Build a Classifier Using either the K-means or K-medoids Clustering Algorithm
    classifier = build_classifier(classification, clusters)

    # # Fit Classifier to tf*idf data (create multi-dimensional clusters)
    classifier.fit_predict(tfidf_vectors)

    # Reduce Dimensionality of Clusters
    data_2D = reduce_dimensions(tfidf_vectors, d_reduce, classifier, n_components, learning_rate, init, perplexity)

    centres_matrix = None
    twoD_matrix = data_2D[0]
    if d_reduce == "PCA":
        centres_matrix = data_2D[1]

    # Plot the 2D Matrix
    fig, (plot) = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(10, 10))
    draw_subplots(plot, f"Colours = {d_reduce} Clusters\nShapes = Scribal Hands", twoD_matrix,
                  classifier.labels_, f"{d_reduce} Cluster", hand_numbers, hand_names)

    # fig, (plot1, plot2, plot3) = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(20, 10))
    # draw_subplots(plot1, 'Colours = Author Clusters', twoD_matrix,
    #               classifier.labels_, "Potential Author", centres=centres_matrix)
    # draw_subplots(plot2, 'Colours = Scribal Hands', twoD_matrix,
    #               hand_numbers, hand_names)
    # draw_subplots(plot3, 'Colours = Scribal Hands\nShapes = Clusters', twoD_matrix,
    #               hand_numbers, hand_names, classifier.labels_)

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
