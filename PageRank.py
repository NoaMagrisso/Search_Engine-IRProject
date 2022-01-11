import json

import pandas as pd

class PageRank:

    def __init__(self):
        self.page_rank_df = pd.read_csv('PageRank.csv', header=None)
        self.page_rank_df.columns = ["id", "PageRank"]
        self.page_rank_df.set_index('id', inplace=True)

    def get_page_rank(self, list_of_doc_id):
        page_rank_list = []
        for doc_id in list_of_doc_id:
            page_rank_list.append(self.page_rank_df.loc[doc_id])
        return page_rank_list


# PageRank().get_page_rank()

