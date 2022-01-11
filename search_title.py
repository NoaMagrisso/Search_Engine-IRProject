import math

import numpy as np
import pandas as pd

from contextlib import closing
from inverted_index_gcp import MultiFileReader
bucket_name = "205963135"

def read_posting_list(inverted, w):
    TUPLE_SIZE = 6
    TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
    with closing(MultiFileReader(bucket_name)) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, "title")
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
    return posting_list


def get_posting_gen(index, query):
    """
    This function returning the generator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    w_pls_dict = {}
    for term in query:
        temp_pls = read_posting_list(index, term)
        w_pls_dict[term] = temp_pls
    return w_pls_dict


def generate_relevant_dict(query_to_search, index):
    dict = {}
    w_pls_dict = get_posting_gen(index, query_to_search) #TODO change to bring only relevant words
    for term in query_to_search:
        if term in w_pls_dict.keys():
            for doc_id, tf in w_pls_dict[term]:
                dict[doc_id] = dict.get(doc_id, 0) + 1
    return dict


def get_candidate_documents_and_scores(query_to_search, index, words, pls, DL):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(DL)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq / DL[doc_id]) * math.log(N / index.df[term], 10)) for doc_id, freq in
                               list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls, DL):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(np.unique(query_to_search))
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words, pls,
                                                           DL)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = query_to_search

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------`
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    cosine_sim_dict = {}
    # YOUR CODE HERE
    q_size = np.linalg.norm(Q)
    for doc_id, row in D.iterrows():
        top = np.dot(row, Q)
        di_size = np.linalg.norm(row.to_numpy())
        cosine_sim_dict[doc_id] = top / (di_size * q_size)
    return cosine_sim_dict


def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]
