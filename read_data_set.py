import pandas as pd
# import stanza
import nltk
from nltk.corpus import treebank

def read_data(file):
    data = pd.read_csv(file, sep="\t", header=None, skiprows=1)
    data.columns = ['date', 'text', 'emotion', 'level']
    print(data.axes)

    texts = [[] for x in range(4)]

    for index, row in data.iterrows():
        texts[int(row['level'][0])].append(row['text'])
    
    return texts


tweets= read_data('data_test.txt')
# stanza.download('en') 
# nlp = stanza.Pipeline('en')
# doc = nlp(tweets[0])
# doc.sentences[0].print_dependencies()

def get_bag_of_words(tweets):
    bag_of_words= tweets.copy()
    for level in range(len(tweets)):
        for twt in range(len(tweets[level])):
            bag_of_words[level][twt]= nltk.word_tokenize(tweets[level][twt])
    return bag_of_words



print(get_bag_of_words(tweets))



