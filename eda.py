



import re
#imported the data
import pandas as pd
data=pd.read_csv("D:/NLP project/qa_Electronics.csv")

#dropped teh data
data.drop("unixTime",axis=1,inplace=True)
data.columns

#importing the stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
#importing the duplicate the stopwords
from wordcloud import WordCloud, STOPWORDS
newStopWords = set(STOPWORDS)
stop.extend(newStopWords)
stop.extend(["yes","one","use","bought","got","put","using","still","turn","kind","really","take","","thank","work","well","better","make","see","going","hold","though","either","two","look","good","look","without","please","let","know","im","look","want","anyone","come","need","thank","use","say"])
#now you have added all the words of STOPWORDS and other words in stop with help of extend function
#now we have created the stop function
stop

#here we are created the contraction_dict in dictionary format
contractions_dict = {
     'didn\'t': 'did not',
     'don\'t': 'do not',
 }

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

#we are creating the expand_contractions to identify
def expand_contractions(s, contractions_dict=contractions_dict):
     def replace(match):
         return contractions_dict[match.group(0)]
     return contractions_re.sub(replace, s)
#cleaning the data by removing the punctuation
def clean_text_round1(text):
    text=text.lower()
    text=expand_contractions(text)#remove contractions
    text=re.sub(r"http\S+", "", text)#remove urls
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\w*\d\w*','',text)
    text=re.sub('[''""]','',text)
    text=re.sub('\n','',text)
    text=re.sub(' x ','',text)
    return text
round1= lambda x: clean_text_round1(str(x))
#applying the cleaning for question and answer in data
import string
qdata=pd.DataFrame(data.question.apply(round1))
adata=pd.DataFrame(data.answer.apply(round1))
data.head()

#applying the stop words in question and answer in data.
qdata["question_with_stopwords"]=qdata["question"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
adata["answer_with_stopwords"]=adata["answer"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

qdata.question_with_stopwords.head()

#applying lemmatization on question_with_stopwordsand answer_with_stopwords
import nltk
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w,'v') for w in w_tokenizer.tokenize(str(text))]

qdata['question_with_lem']=pd.DataFrame(qdata['question_with_stopwords'])
adata["answer_with_lem"]=pd.DataFrame(adata["answer_with_stopwords"])
adata["answer_with_lem"]=adata.answer_with_stopwords.apply(lemmatize_text)
qdata["question_with_lem"]=qdata.question_with_stopwords.apply(lemmatize_text)

#applying stemmer in data
from nltk.stem import PorterStemmer
pst = PorterStemmer()
qdata["question_with_stemmer"]=qdata.question_with_lem.apply(lambda x: ' '.join([pst.stem(y) for y in x]))
adata["answer_with_stemmer"]=adata.answer_with_lem.apply([lambda x: ' '.join([pst.stem(y) for y in x])])

qdatastem=qdata.question_with_stemmer.tolist()
adatastem=adata.answer_with_stemmer.tolist()

#Question and Answer WordCloud
import matplotlib.pyplot as plt
wordcloud_question = WordCloud(width=2800,height=2400).generate(str(qdatastem))
plt.imshow(wordcloud_question)
wordcloud_answer = WordCloud(width=2800,height=2400).generate(str(adatastem))
plt.imshow(wordcloud_answer)

#doing question and answer on positive word

with open("D:/data science/text mining/positive-stopwords.txt","r") as pos:
  poswords = pos.read().split("\n")


with open("D:/data science/text mining/negative-stopwords.txt","r") as neg:
  negwords = neg.read().split("\n")
 
qposwords = " ".join ([w for w in qdatastem if w in poswords])
qnegwords = " ".join ([w for w in qdatastem if w in negwords])

wordcloud_question_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate((qposwords))
plt.imshow(wordcloud_question_pos)

wordcloud_question_neg= WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate((qnegwords))
plt.imshow(wordcloud_question_neg)

aposwords= " ".join([w for w in adatastem if w in poswords])
anegwords=" ".join([w for w in adatastem if w in negwords])

wordcloud_answer_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate((aposwords))
plt.imshow(wordcloud_answer_pos)

wordcloud_answer_pos = WordCloud(
                      background_color="white",
                      width=1800,
                      height=1400
                     ).generate((anegwords))
plt.imshow(wordcloud_answer_pos)



#(or)

# Python3 code to demonstrate 
# Bigram formation 
# using list comprehension + enumerate() + split() 
   
# initializing list  
test_list = ['geeksforgeeks is best', 'I love it'] 
  
# printing the original list  
print ("The original list is : " + str(test_list)) 
  
# using list comprehension + enumerate() + split() 
# for Bigram formation 
res = [(x, i.split()[j + 1]) for i in test_list  
       for j, x in enumerate(i.split()) if j < len(i.split()) - 1] 
  
# printing result 
print ("The formed bigrams are : " + str(res)) 


###############33

from sklearn.feature_extraction.text import CountVectorizer
corpus=qdata["question_with_stemmer"]

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]
#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=25)
top_df = pd.DataFrame(top_words)
top_df.columns=["Word", "Freq"]
top_df.head(25)
#Barplot of most freq words
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

#Most frequently occuring Bi-grams
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
top2_words = get_top_n2_words(corpus, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)
#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)

#Most frequently occuring Tri-grams
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]
top3_words = get_top_n3_words(corpus, n=20)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]
print(top3_df)
#Barplot of most freq Tri-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
j.set_xticklabels(j.get_xticklabels(), rotation=45)







adata["answer_with_stemmer"]
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

sent = preprocess(str(adata.answer_with_stemmer))
sent


pattern = 'NP: {<DT>?<JJ>*<NN>}'    

#Using this pattern, we create a chunk parser and test it on our sentence.   

cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs) 
cs.draw()    
qdata.columns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#now we are doing tfidf for single word with stemmind data
tvec_ta = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
tvec_weights_ta = tvec_ta.fit_transform(qdata.question_with_stemmer.dropna())
weights_ta = np.asarray(tvec_weights_ta.mean(axis=0)).ravel().tolist()
weights_df_ta = pd.DataFrame({'term': tvec_ta.get_feature_names(), 'weight': weights_ta})
weights_df_ta=weights_df_ta.sort_values(by='weight', ascending=False)
weights_df_ta.to_csv(r"D:/rough work\question.csv", index = None, header=True)

adata.columns
#now we are doing tfidf for single word with stemmind data
tvec_ta = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
tvec_weights_ta = tvec_ta.fit_transform(adata.answer_with_stemmer.dropna())
weights_ta = np.asarray(tvec_weights_ta.mean(axis=0)).ravel().tolist()
weights_df_ta = pd.DataFrame({'term': tvec_ta.get_feature_names(), 'weight': weights_ta})
weights_df_ta=weights_df_ta.sort_values(by='weight', ascending=False)
weights_df_ta.to_csv(r"D:/rough work\answer.csv", index = None, header=True)

