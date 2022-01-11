import concurrent.futures
import csv
import json
import pickle
from time import time

from flask import Flask, request, jsonify
from google.cloud import storage

import evaluations
from BM25 import BM25, merge_results
from PageRank import PageRank
from Tokenizer import Tokenizer
from search_body import *
from search_title import *
from search_anchor import *


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        self.tokenizer = Tokenizer()
        self.page_rank = PageRank()
        bucket_name = "205963135"
        client = storage.Client()
        self.my_bucket = client.bucket(bucket_name=bucket_name)
        for blob in client.list_blobs(bucket_name):
            if blob.name == "postings_gcp_body/body/index.pkl":
                with blob.open('rb') as openfile:
                    self.index_body = pickle.load(openfile)
            elif blob.name == "postings_gcp_body_stem/index.pkl":
                with blob.open('rb') as openfile:
                    self.index_body_stemmed = pickle.load(openfile)
            elif blob.name == "postings_gcp_title/index.pkl":
                with blob.open('rb') as openfile:
                    self.index_title = pickle.load(openfile)
            elif blob.name == "postings_gcp_title_stem/index.pkl":
                with blob.open('rb') as openfile:
                    self.index_title_stemmed = pickle.load(openfile)
            elif blob.name == "postings_gcp_anchor/index.pkl":
                with blob.open('rb') as openfile:
                    self.index_title_stemmed = pickle.load(openfile)
        self.executor = concurrent.futures.ThreadPoolExecutor(2)
        with open('PageRank.csv', mode='r') as infile:
            reader = csv.reader(infile)
            self.page_rank_log = {int(rows[0]): math.log(float(rows[1]), 10) for rows in reader}
        with (open('IdTitle.pkl', 'rb')) as openfile:
            try:
                self.id_title = pickle.load(openfile)
            except EOFError:
                pass
        with (open('pageviews.pkl', 'rb')) as openfile:
            try:
                self.page_views = pickle.load(openfile)
            except EOFError:
                pass
        with (open('normalized_doc_vec.pkl', 'rb')) as openfile:
            try:
                self.normalized_doc_vec_dict = pickle.load(openfile)
            except EOFError:
                pass
        with (open('DL_body.pkl', 'rb')) as openfile:
            try:
                self.DL_body = pickle.load(openfile)
            except EOFError:
                pass
        with (open('DL_title.pkl', 'rb')) as openfile:
            try:
                self.DL_title = pickle.load(openfile)
            except EOFError:
                pass
        print("body index BM25 started")
        t_start = time()
        self.BM25_body = BM25(self.index_body_stemmed, self.DL_body, "body_stem", self.page_rank_log, k1=1.4, b=0.6)
        pr_time = time() - t_start
        print("body index finished in " + str(pr_time))
        print("title index BM25 started")
        t_start = time()
        self.BM25_title = BM25(self.index_title_stemmed, self.DL_title, "title_stem", self.page_rank_log, k1=1.4, b=0.6)
        pr_time = time() - t_start
        print("title index finished in " + str(pr_time))
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    print("search started")
    t_start = time()
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query_tokens = list(set(app.tokenizer.tokenize(query, True)))
    future_body = app.executor.submit(app.BM25_body.search, {1: query_tokens}, 100)
    future_title = app.executor.submit(app.BM25_title.search, {1: query_tokens}, 100)
    dict_merged = merge_results(future_body.result(), future_title.result(), title_weight=0.4, text_weight=0.6, N=100)
    res = dict_merged[1]
    res = [(str(doc_id), app.id_title[doc_id]) for doc_id, score in res]
    # END SOLUTION
    pr_time = time() - t_start
    print("search finished in " + str(pr_time))
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # res.append(get_topN_score_for_queries({1: app.tokenizer.tokenize(query,False)}, app.index_body, app.DL_body, N=100))
    res_list = get_topN_score_for_queries(app.tokenizer.tokenize(query, False), app.index_body, app.DL_body, app.normalized_doc_vec_dict, N=100)
    res_list = [(doc_id, app.id_title[doc_id]) for doc_id, score in res_list]
    res = res_list
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    # res.append(get_topN_score_for_queries({1: query.split()}, app.index_title, app.DL_title, N=100))
    dict = generate_relevant_dict(app.tokenizer.tokenize(query, False), app.index_title)
    res = sorted([(doc_id, score) for doc_id, score in dict.items()], key=lambda x: x[1], reverse=True)
    res = [(doc_id, app.id_title[doc_id]) for doc_id, score in res]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    dict = generate_relevant_dict_anchor(app.tokenizer.tokenize(query, False), app.index_anchor)
    res = sorted([(doc_id, score) for doc_id, score in dict.items()], key=lambda x: x[1], reverse=True)
    res = [(doc_id, app.id_title[doc_id]) for doc_id, score in res]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    # import requests
    # requests.post('http://127.0.0.1:8080//get_pagerank', json=[1, 5, 8])
    try:
        res = []
        wiki_ids = request.get_json(force=True)
        if len(wiki_ids) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        res = app.page_rank.get_page_rank(wiki_ids)
        # END SOLUTION
        return jsonify(res)
    except Exception as e:
        print(e)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    try:
        res = []
        wiki_ids = request.get_json()
        if len(wiki_ids) == 0:
            return jsonify(res)
        # BEGIN SOLUTION
        for doc_id in wiki_ids:
            res.append(app.page_views[doc_id])
        # END SOLUTION
        return jsonify(res)
    except Exception as e:
        print(e)


@app.route("/evaluate")
def get_evaluate():
    parameters_list = [ (1.4, 0.6)]
    weights_list = [(0.4, 0.6), (0.3, 0.7)]
    N = 40
    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)
    query_dict = {}
    for index, query in enumerate(queries.keys()):
        query_dict[index] = [token for token in app.tokenizer.tokenize(query, True)]
    grid_models = evaluations.grid_search_models(query_dict, dict(enumerate(queries.values())), app.DL_body, "body_stem",
                                                 app.DL_title, "title_stem", parameters_list, weights_list, N,
                                                 app.index_title_stemmed, app.index_body_stemmed, app.page_rank_log)
    k1, b, title_w, body_w = max(grid_models, key=grid_models.get)
    print(k1, b, title_w, body_w)
    bm25_title = BM25(app.index_title_stemmed, app.DL_title, "title_stem", app.page_rank_log, k1=k1, b=b)
    bm25_text = BM25(app.index_body_stemmed, app.DL_body, "body_stem", app.page_rank_log, k1=k1, b=b)
    bm25_queries_score_test_title = bm25_title.search(query_dict, N=N)
    bm25_queries_score_test_text = bm25_text.search(query_dict, N=N)
    merge_res = merge_results(bm25_queries_score_test_title, bm25_queries_score_test_text, title_weight=title_w,
                              text_weight=body_w, N=N)
    test_metrices = evaluations.evaluate(dict(enumerate(queries.values())), merge_res, k=N, print_scores=False)
    print(test_metrices['MAP@k'])
    res = []
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
