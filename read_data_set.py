import pandas as pd
# import stanza
import nltk
from nltk.corpus import treebank
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def read_data(file):
    data = pd.read_csv(file, sep="\t", header=None, skiprows=1)
    data.columns = ['date', 'text', 'emotion', 'level']
    print(data.axes)

    texts = [[] for x in range(4)]

    for index, row in data.iterrows():
        texts[int(row['level'][0])].append(row['text'])
        # texts[int(row['level'][0])]+=' ' +row['text'])
    
    return texts


# tweets= read_data('2018-EI-oc-En-sadness-dev.txt')

tweets= read_data('data_test.txt')
print(tweets[2])

def get_bag_of_words(tweets):
    bag_of_words= tweets.copy()
    for level in range(len(tweets)):
        for twt in range(len(tweets[level])):
            bag_of_words[level][twt]= nltk.word_tokenize(tweets[level][twt])
    return bag_of_words


# print(get_bag_of_words(tweets))

 
 
# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True)
 
# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(tweets[2])

# get the first vector out (for the first document)
first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[1]

df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df=df.sort_values(["tfidf"], ascending=False)

print(df)
# print(df.at['🙈', "tfidf"])

