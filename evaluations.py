from BM25 import merge_results, BM25


# import matplotlib.pyplot as plt


def intersection(l1, l2):
    """
    This function perform an intersection between two lists.

    Parameters
    ----------
    l1: list of documents. Each element is a doc_id.
    l2: list of documents. Each element is a doc_id.

    Returns:
    ----------
    list with the intersection (without duplicates) of l1 and l2
    """
    return list(set(l1) & set(l2))


def precision_at_k(true_list, predicted_list, k=40):
    """
    This function calculate the precision@k metric.

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    mone = len([x for x in predicted_list[:k] if x in true_list])
    mechane = k
    precision = round(mone / float(mechane), 3)
    return precision


def average_precision(true_list, predicted_list, k=40):
    """
    This function calculate the average_precision@k metric.(i.e., precision in every recall point).

    Parameters
    -----------
    true_list: list of relevant documents. Each element is a doc_id.
    predicted_list: sorted list of documents predicted as relevant. Each element is a doc_id. Sorted is performed by relevance score
    k: integer, a number to slice the length of the predicted_list

    Returns:
    -----------
    float, average precision@k with 3 digits after the decimal point.
    """
    # YOUR CODE HERE
    sum = 0.0
    div = 0
    for x in range(1, k + 1):
        if len(predicted_list) >= x and predicted_list[x - 1] in true_list:
            sum += precision_at_k(true_list, predicted_list, x)
            div += 1
    if div == 0:
        return 0
    return round(sum / div, 3)


def evaluate(true_relevancy, predicted_relevancy, k, print_scores=False):
    """
    This function calculates multiple metrics and returns a dictionary with metrics scores across different queries.
    Parameters
    -----------
    true_relevancy: list of tuples indicating the relevancy score for a query. Each element corresponds to a query.
    Example of a single element in the list:
                                            (3, {'question': ' what problems of heat conduction in composite slabs have been solved so far . ',
                                            'relevance_assessments': [(5, 3), (6, 3), (90, 3), (91, 3), (119, 3), (144, 3), (181, 3), (399, 3), (485, 1)]})

    predicted_relevancy: a dictionary of the list. Each key represents the query_id. The value of the dictionary is a sorted list of relevant documents and their scores.
                         The list is sorted by the score.
    Example:
            key: 1
            value: [(13, 17.256625), (486, 13.539465), (12, 9.957595), (746, 9.599499999999999), (51, 9.171265), .....]

    k: integer, a number to slice the length of the predicted_list

    print_scores: boolean, enable/disable a print of the mean value of each metric.

    Returns:
    -----------
    a dictionary of metrics scores as follows:
                                                        key: metric name
                                                        value: list of metric scores. Each element corresponds to a given query.
    """

    avg_precision_lst = []

    metrices = {
        'MAP@k': avg_precision_lst,
    }

    for query_id, query_info in true_relevancy.items():
        predicted = [doc_id for doc_id, score in predicted_relevancy[query_id]]
        ground_true = [doc_id for doc_id in query_info]
        avg_precision_lst.append(average_precision(ground_true, predicted, k=k))

    if print_scores:
        for name, values in metrices.items():
            print(name, sum(values) / len(values))

    return metrices


def grid_search_models(data, true_relevancy, DL_body, index_type_body, DL_title, index_type_title, bm25_param_list,
                       w_list, N, idx_title, idx_body, page_rank_log):
    """
      This function is performing a grid search upon different combination of parameters.
    The parameters can be BM25 parameters (i.e., bm25_param_list) or different weights (i.e., w_list).

    Parameters
    ----------
    data: dictionary as follows:
                            key: query_id
                            value: list of tokens

    true_relevancy: list of tuples indicating the relevancy score for a query. Each element corresponds to a query.
    Example of a single element in the list:
                                            (3, {'question': ' what problems of heat conduction in composite slabs have been solved so far . ',
                                            'relevance_assessments': [(5, 3), (6, 3), (90, 3), (91, 3), (119, 3), (144, 3), (181, 3), (399, 3), (485, 1)]})

    bm25_param_list: list of tuples. Each tuple represent (k,b1) values.

    w_list: list of tuples. Each tuple represent (title_weight,body_weight).
    N: Integer. How many document to retrieve.

    idx_title: index build upon titles
    idx_body:  index build upon bodies
    ----------
    return: dictionary as follows:
                            key: tuple indiciating the parameters examined in the model (k1,b,title_weight,body_weight)
                            value: MAP@N score
    """
    models = {}
    # YOUR CODE HERE
    for k, b in bm25_param_list:
        for w_t, w_b in w_list:
            key = (k, b, w_t, w_b)
            bm25_title = BM25(idx_title, DL_title, index_type_title, page_rank_log, k1=k, b=b)
            bm25_body = BM25(idx_body, DL_body, index_type_body, page_rank_log, k1=k, b=b)
            predicted_title = bm25_title.search(data, N=N)
            predicted_body = bm25_body.search(data, N=N)
            merged = merge_results(predicted_title, predicted_body, w_t, w_b, N=N)
            map_res = 0
            count_of_ap = 0
            for query_id in range(len(merged)):
                predicted = [x[0] for x in merged[query_id]]
                true_list = [x for x in true_relevancy[query_id]]
                ap = average_precision(true_list, predicted, N)
                if ap > 0:
                    count_of_ap += 1
                    map_res += ap
            models[key] = map_res / count_of_ap
            print("The query is: " + str(data))
            print("(k, b, w_t, w_b)=" + str(key))
            print("The MAP@40 is: " + str(map_res / count_of_ap))
    return models
