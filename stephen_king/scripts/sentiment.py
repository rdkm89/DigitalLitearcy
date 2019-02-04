# Import libraries
import pandas as pd
import os,sys
from os import listdir
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from nltk.stem.snowball import SnowballStemmer
        
# Define functions
def list_textfiles(directory):
        """Return a list of filenames ending in '.txt' in DIRECTORY."""
        textfiles = []
        for filename in listdir(directory):
            if filename.endswith(".txt"):
                textfiles.append(filename)
        return textfiles

def re_nalpha(str):
    pattern = re.compile(r'[^\w\s]', re.U)
    return re.sub(r'\n','',re.sub(r'_', '', re.sub(pattern, '', str)))

def word_tokenize(text):
        return re.findall(r'\w+', text.lower())

def tokenizer(text,size=1):
	# tokenize to 1 gram strings
	result_size1 = word_tokenize(text)
	if size == 1:
		return result_size1
	# tokenize into n gram strings
	else:
		temp_list = []
		temp_size = 0
		for i in range(0,len(result_size1)-size+1):
			temp_str=''
			for j in range(size):
				temp_str = temp_str+" "+ result_size1[temp_size+j]
			temp_list.append(temp_str[1:])
			temp_size += 1
		return temp_list

def stemmer(wordList):
     """ Stem using Snowball """
     stemmer = SnowballStemmer("english")
     return [stemmer.stem(tempWord) for tempWord in wordList]

def slice_tokens(tokens, n = 10, cut_off = True):
        """ Create chunks of 10 words at a time """
        slices = []
        for i in range(0,len(tokens),n):
                slices.append(tokens[i:(i+n)])
        if cut_off:
                del slices[-1]
        return slices

def analyze(text):
        """ Assign sentiment score using AFINN"""
        sentimentScore = []
        for temp in text:
                sentimentScore.append(sum([afinn.get(token,0.0) for token in temp]))
        return sentimentScore

def func(x, a, b, c,d,e):
        """ Smoother help function """
        return a*x + b*x*x + c*x*x*x +d*x*x*x*x +e

def smoother(dump):
        """ Rolling average to smooth """
        myInd = np.arange(len(dump))
        popt, pcov = curve_fit(func, myInd, dump)
        return [func(i,*popt) for i in myInd]


def visualise_sentiments(text):
        lines = f.readlines()
        content = ' '.join(lines)

        nalpha_content = re_nalpha(content)
        tokenized_content = tokenizer(nalpha_content)
        stemmed_content = stemmer(tokenized_content)
        sliced_content = slice_tokens(tokenized_content)
        score = analyze(sliced_content)
        smoothedScore = smoother(score)

        x = np.arange(len(smoothedScore))

        plt.plot(x, smoothedScore)
        plt.legend(('smoothed'))
        plt.savefig("/Users/au564346/Desktop/" + filenames[i] + ".png")
        plt.close()


if __name__ == "__main__":
        # Read in sentiment dictionary
        afinn = pd.read_csv('afinn.txt', delimiter="\t").set_index('word').to_dict()['sentiment']
        # Path to novels
        CORPUS_PATH = "/Users/au564346/Documents/research/digital_literacies/projects/mathias/cleaned_texts"
        filenames = list_textfiles(CORPUS_PATH)
        files_with_path = [os.path.join(CORPUS_PATH, fn) for fn in filenames]
        #Plot
        i = 0
        for novel in files_with_path:
                with open(novel, encoding ="utf-8") as f:
                        visualise_sentiments(novel)
                i+=1
