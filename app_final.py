import os
os.chdir("D:/NLP project/information retrieval")

from flask import Flask, request, render_template
import pandas as pd
from whoosh.fields import Schema, TEXT
from whoosh import index
from whoosh import qparser
from whoosh import scoring


app = Flask(__name__)

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/finalresult',methods = ['POST'])
def finalresult():
    if request.method == 'POST':
            #search query 
            query = request.form['QA']
            print(query)
            results = []
            ix = index.open_dir("qadata_Index")
            schema = ix.schema
            # Create query parser that looks through designated fields in index
            og = qparser.OrGroup.factory(0.9)
            mp = qparser.MultifieldParser(['question', 'answer'], schema, group = og)
            # This is the user query
            q = mp.parse(request.form['QA'])
            # Actual searcher, prints top 10 hits
            with ix.searcher() as s:
                results = s.search(q, limit = 5)
                for i in range(5):
                    print(results[i]['question'], str(results[i].score), results[i]['answer'])
                return render_template("result.html",searchquery=request.form['QA'],
                                       Q1=results[0]['question'],A1=results[0]['answer'],
                                       Q2=results[1]['question'],A2=results[1]['answer'],
                                       Q3=results[2]['question'],A3=results[2]['answer'],
                                       Q4=results[3]['question'],A4=results[3]['answer'],
                                       Q5=results[4]['question'],A5=results[4]['answer'])

if __name__ == '__main__':
   app.run(debug = True)