import pandas as pd
from whoosh.fields import Schema, TEXT
from whoosh import index
import os, os.path
from whoosh import qparser
#Below module contains implementations of various scoring algorithms.Default is BM25F
#from whoosh import scoring
#Load the data
qadata=pd.read_csv("D:/NLP project/information retrieval/qa_Electronics.csv")
#update the null values answer field with default value
qadata["answer"].fillna("Please Provide more information", inplace = True)
#Schema is created to index on question and answer fields
schema = Schema(question = TEXT (stored = True,  field_boost = 2.0),
                answer = TEXT (stored = True,  field_boost = 2.0))
#Functions to create index for the search fields
def add_stories(i, dataframe, writer):   
    writer.update_document(question = str(dataframe.loc[i, "question"]),
                           answer = str(dataframe.loc[i, "answer"]))
# create and populate index
def populate_index(dirname, dataframe, schema):
    # Checks for existing index path and creates one if not present
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    print("Creating the Index")
    ix = index.create_in(dirname, schema)
    with ix.writer() as writer:
        # Imports stories from pandas df
        print("Populating the Index")
        for i in dataframe.index:
            add_stories(i, dataframe, writer)
#Populate index of the csv file
populate_index("qadata_Index", qadata, schema)
#Query search based on index
def index_search(dirname, search_fields, search_query):
    ix = index.open_dir(dirname)
    schema = ix.schema
    # Create query parser that looks through designated fields in index
    og = qparser.OrGroup.factory(0.9)
    mp = qparser.MultifieldParser(search_fields, schema, group = og)
    # This is the user query
    q = mp.parse(search_query)
    # Actual searcher, prints top 10 hits
    with ix.searcher() as s:
        results = s.search(q, limit = None)
        print("Total Documents: ",ix.doc_count_all())
        print("Retrieved Documents: ",results.estimated_length())
        print(results._get_scorer())
        for i,result in enumerate(results[0:5]):
            print("Search Results: ",result.rank,"Score: ",result.score)
            print("Question: ",result['question'])
            print("Answer: ",result['answer'])
            print("------------------------")
        
#Testing
index_search("qadata_Index", ['question', 'answer'], u"samsung galaxy tab")
