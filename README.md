# Search Engine on Wikipedia
By this project we implemented a search engine on Wikipedia, as we asked to do under the "Information Retreive" course at BGU.
* **Our engine supports in returning a result by _1.6 seconds_ for an average and we can commit _71.42%_ of successful searching.**

## The Code Structure

| Class | Description | Major Methods |
| --- | --- | --- |
| Search_frotened_run | here the search is created. | search func (also calculate the BM25 and Stemming, it's time complexity is measured), Search Body Func (search on the bodies of the all documents of the corpus, which are relevant to the query), Search Title Func (the same as the body func on the titles of the documents) and Search Anchor Func (search on the anchors of each document, while the most intresting is the page who is pointered by other), Page Rank Func (get a list of pageranks and return the page rank for a doc), Page View Func (get a list of pagevies and return the page view for a doc) |
| BM25 | here the calculate of the BM25 measurement | it is used for calculating the BM25 score for each document by k and b parameters |
| Tokenizer | it's has the all tokens after cleaning things, such as stopwords |
| Inverted Index Gcp | a class we got | By changing it a little we found 


| Pickle File's Name | The Information Inside |
| --- | --- |
| ID_Title | a dictionary which it's keys are doc_id, and the values are the titls' doc correspondingly |
| PageView | a dictionary which it's keys are doc_id, and the values are the page view of the document |
| DL_Title | a dictionary which it's keys are doc_id, and the values are length of the doc's title |
| DL_Body | a dictionary which it's keys are doc_id, and the values are length of the doc's body |


![This is an image](https://github.com/NoaMagrisso/Search_Engine-IRProject/blob/main/README_image.JPG)
