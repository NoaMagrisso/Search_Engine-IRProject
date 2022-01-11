# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import pickle
from time import time

import pandas as pd
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    page_rank_df = pd.read_csv('PageRank.csv', header=None)
    page_rank_df.columns = ["id", "PageRank"]
    page_rank_df.set_index('id', inplace=True)
    # importing the module
    syns = wordnet.synsets("google")
    # print(syns)
    # print(syns[0].lemmas()[0].name())
    # t_start = time()
    # synonyms = []
    # antonyms = []
    # token_list = ["best", "marvel", "movie"]
    # for token in token_list:
    #     for syn in wordnet.synsets(token)[:5]:
    #         for l in syn.lemmas():
    #             if l.name().lower() not in synonyms and l.name().lower() not in token_list:
    #                 synonyms.append(l.name().lower())
    # print(synonyms)
    # print(len(synonyms))
    # pr_time = time() - t_start
    # print("title index finished in " + str(pr_time))
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
