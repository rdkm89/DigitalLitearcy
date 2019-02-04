#!/usr/bin/env python3 -W ignore
# Import packages
print("importing packages...")
import warnings
import glob
import collections
from tqdm import tqdm
import numpy as np
import pandas as pd
import spacy
import gensim
from gensim.models import Phrases
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from stop_words import get_stop_words
import pyLDAvis.gensim
import matplotlib.pyplot as plt

# intialise functions
print("defining functions...")
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def get_lda_topics(model, num_topics):
    word_dict = {};
    for i in range(num_topics):
        words = model.show_topic(i, topn = 20);
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in words];
    return pd.DataFrame(word_dict)

# Define function to present topics in a neater format
def explore_topic(lda_model, topic_number, topn, output=True):
    """
    accept a ldamodel, atopic number and topn vocabs of interest
    prints a formatted list of the topn terms
    """
    terms = []
    for term, frequency in lda_model.show_topic(topic_number, topn=topn):
        terms += [term]
        if output:
            print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))
    return terms

# Stopwords
print("creating stopwords...")
stop_1 = get_stop_words("english")
stop_2 = stopwords.words('english')
stopwords = list(set(stop_1 + stop_2))
stopwords.extend(["-PRON-"])

warnings.filterwarnings('ignore')
# Main Function
if __name__ == "__main__":    
    # Initialise spaCy
    print("intialising spaCy...")
    nlp = spacy.load('en')

    # Read from csv
    """
    TODO: write script that concatenates all journals from different sources.
    """
    print("reading data...")
    allFiles = glob.glob("data/elsevier/*.csv")
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)
        data = pd.concat(list_, axis = 0, ignore_index = True)

    # Document preprocesing
    # Main pipeline
    print("processing abstracts in batches of 100...")
    processed_docs = []    
    for doc in tqdm(nlp.pipe(data["abstract"], n_threads=3, batch_size=100), total=len(data)):
        # Named entities.
        ents = doc.ents
        # Keep only words (no numbers, no punctuation).
        # Lemmatize tokens, remove punctuation and remove spaCy stopwords.
        doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        # Remove common words from a stopword list.
        doc = [token for token in doc if token not in stopwords]
        # Add named entities, but only if they are a compound of more than word. E.g. - United_States
        doc.extend([str(entity) for entity in ents if len(entity) > 1])
        processed_docs.append(doc)
    # Rename and delete
    docs = processed_docs
    del processed_docs

    print("calculating bigrams and trigrams...")
    # Compute bigrams & trigrams
    bigram = Phrases(docs, min_count=5)
    trigram = Phrases(bigram[docs])
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
        for token in trigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # Create a dictionary representation of the documents, and filter out frequent and rare words.
    print("creating dictionary...")
    dictionary = Dictionary(docs)
    # Remove rare and common tokens.
    max_freq = 0.8
    min_wordcount = 10
    dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)
    # This sort of "initializes" dictionary.id2token.
    _ = dictionary[0]

    # Vectorize data.
    print("vectorizing...")
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    # Inspect elements
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    # Set training parameters.
    num_topics = 12
    update_every = 1
    chunksize = 100
    passes = 10
    iterations = 100
    eval_every = 1

    print("running multiple iterations for maximal coherence...")
    # Multiple random initializations
    model_list = []
    for i in range(5):
        model = tqdm(gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, 
                    update_every=update_every, chunksize=chunksize, passes=passes, eval_every=eval_every))
        top_topics = model.top_topics(corpus)
        tc = sum([t[1] for t in top_topics])
        model_list.append((model, tc))
    # Choose most coherent model and display coherence
    model, tc = max(model_list, key=lambda x: x[1])

    print("printing topic summaries...")
    topic_summaries = []
    for i in range(num_topics):
        print('Topic '+str(i)+' |---------------------\n')
        tmp = explore_topic(model,topic_number=i, topn=10, output=True )
        #     print tmp[:5]
        topic_summaries += [tmp[:5]]
    print(topic_summaries)

    print("creating pyLDAVis...")
    # Inspect model in pyLDAvis
    viz = pyLDAvis.gensim.prepare(model, corpus, dictionary, sort_topics=False)
    # Inspect model in pyLDAvis
    pyLDAvis.save_html(viz, 'output/lda.html')
    print("done!")
