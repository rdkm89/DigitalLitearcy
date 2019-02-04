"""
NER for sermons in content.dat

This is spaghetti code used for generating the tables of NER data used in the project.
It can defintely be optimised and cleaned up a good deal. I'd actually like to modularise
the whole thing at some point. But it works for now.


"""

import pandas as pd
import numpy as np
from nltk.stem.snowball import DanishStemmer
import nltk.data
from polyglot.text import Text


"""
Creating NER outputs

Note: .dat files not included in the Github repo for privacy reasons. 
Contact rdkm@cc.au.dk for more information.

Script adapted from earlier work by K.Nielbo.

"""
DF = pd.read_csv("content.dat", header = 0, index_col = None)

content = DF["content"].tolist()
fnames = DF["id"].tolist()

tokenizer = nltk.data.load("tokenizers/punkt/norwegian.pickle")
DATA_ne = []
i = 0
for i, text in enumerate(content):
#for i, text in enumerate(content[:4]):
    print("file {}".format(i))
    # sentence disambiguation
    sents = tokenizer.tokenize(text)
    # NER1
    text_entities = []
    for blob in sents:
        textblob = Text(blob, hint_language_code='da')
        text_entities.append(textblob.entities)
        #if textblob.entities:
        #    text_entities.append(textblob.entities)
    DATA_ne.append([fnames[i],text_entities])


DF_pos = pd.DataFrame(DATA_ne)
DF_pos.columns = ["id", "NE"]
DF_pos.to_csv("content_entities.dat", index = False)


"""
The following section extracts all named people in the dataset.

These are then grouped by name and counted to give the total occurence
of each individual.

"""

# Extract each person (I-PER) from the dataset at sentence level
entity_class = "I-PER"
df = DF_pos
entities = df["NE"].tolist()
fname = df["id"]
res = []
for i, doc in enumerate(entities):
    for ii, sent in enumerate(doc):
        if sent:
            for entity in sent:
                if entity.tag == entity_class:
                    res.append([fname[i], ii, (", ".join(entity))])

# create dataframe with sensible column names
people = pd.DataFrame(res)
people.columns = ["fname","sentence", entity_class]

# Clean up any punctuation, whitespace, etc
people["I-PER"] = people['I-PER'].str.replace('[^\w\s]','')
# To lower
people["I-PER"] = people["I-PER"].str.lower()
# Replace any empty cells with NaN and then remove NaN
people["I-PER"].replace('', np.nan, inplace=True)
people = people.dropna()

# Stem remaining names with Snowball
# This is far from perfect but it gets rid of things like possessives
people["I-PER"] = people["I-PER"].map(DanishStemmer.stem(people["I-PER"]))

# Save to csv
people.to_csv("content_{}.csv".format(entity_class), index = False)
# Group entities and count totals for each individual
people_counted = pd.DataFrame(people_cleaned.groupby(['NAMED PERSON']).size().reset_index(name='counts')).sort_values(by='counts', ascending=False)
people_counted = people_counted.counts.sum()
# Save to csv
people_counted.to_csv("content_counted_{}.csv".format(entity_class), index = False)


"""
As above, except this time with named locations (I-LOC)

"""

# """

# extract each location (I-LOC) from the dataset at sentence level
entity_class = "I-LOC"
df = DF_pos
entities = df["NE"].tolist()
fname = df["id"]
res = []
for i, doc in enumerate(entities):
    for ii, sent in enumerate(doc):
        if sent:
            for entity in sent:
                if entity.tag == entity_class:
                    res.append([fname[i], ii, (", ".join(entity))])

# create dataframe with sensible column names
locations = pd.DataFrame(res)
locations.columns = ["fname","sentence", entity_class]

# Clean up any punctuation, whitespace, etc
locations["I-LOC"] = locations["I-LOC"].str.replace('[^\w\s]','')
# To lower
locations["I-LOC"] = locations["I-LOC"].str.lower()
# Replace any empty cells with Nan; remove NaN
locations["I-LOC"].replace('', np.nan, inplace=True)
locations = locations.dropna()
# Save to csv
locations.to_csv("content_{}.csv".format(entity_class), index = False)

# Group locations and count totals for each place
locations_counted = pd.DataFrame(locations.groupby(['I-LOC']).size().reset_index(name='counts')).sort_values(by='counts', ascending=False)
Locations_counted.counts.sum()
# Save to csv
locations_counted.to_csv("content_counted_{}.csv".format(entity_class), index = False)


""" 
Again, same as above but this time with organisations (I-ORG)

"""

# extract each organisation (I-ORG) at sentence level for each document
entity_class = "I-ORG"
df = DF_pos
entities = df["NE"].tolist()
fname = df["id"]
res = []
for i, doc in enumerate(entities):
    for ii, sent in enumerate(doc):
        if sent:
            for entity in sent:
                if entity.tag == entity_class:
                    res.append([fname[i], ii, (", ".join(entity))])

# create dataframe with sensible column names
orgs = pd.DataFrame(res)
orgs.columns = ["fname","sentence", entity_class]

# Clean up any punctuation, whitespace, etc
orgs["I-ORG"] = orgs["I-ORG"].str.replace('[^\w\s]','')
# To lower
orgs["I-ORG"] = orgs["I-ORG"].str.lower()
# Replace any empty cells with Nan; remove NaN
orgs["I-ORG"].replace('', np.nan, inplace=True)
orgs = orgs.dropna()
# Save to csv
orgs.to_csv("content_{}.csv".format(entity_class), index = False)

# Group organisaitons and count totals
orgs_counted = pd.DataFrame(orgs.groupby(['I-ORG']).size().reset_index(name='counts')).sort_values(by='counts', ascending=False)
orgs_counted.counts.sum()
# Save to csv
orgs_counted.to_csv("content_counted_{}.csv".format(entity_class), index = False)

"""
This last section joins named entity counts to the sermon metadata.

NB: metadata not avaiable on Github for privacy reasons. 
Contact rdkm@cc.au.dk for more info.

"""

# read in csvs
df_people = pd.read_csv("content_I-PER.csv")
df_locations = pd.read_csv("content_I-LOC.csv")
df_orgs = pd.read_csv("content_I-ORG.csv")
# read in metadata
meta = pd.read_excel("Joined_Meta.xlsx")

# dataframe merge
ner_people_with_meta = df_people.merge(meta, left_on='fname', right_on='ID-dok', how='left')
ner_locations_with_meta = df_locations.merge(meta, left_on='fname', right_on='ID-dok', how='left')
ner_orgs_with_meta = orgs.merge(meta, left_on='fname', right_on='ID-dok', how='left')

# save all to csv,
ner_people_with_meta.to_csv("ner_people_with_meta.csv", encoding='utf8')
ner_locations_with_meta.to_csv("ner_locations_with_meta.csv", encoding='utf8')
ner_orgs_with_meta.to_csv("ner_orgs_with_meta.csv", encoding='utf8')
