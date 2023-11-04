import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import pyodbc
import networkx as nx
import matplotlib.pyplot as plt
import re
import spacy
from nltk import SnowballStemmer
import gensim
import Levenshtein as lev
from sentence_transformers import SentenceTransformer
import itertools
from collections import Counter, OrderedDict
from wordcloud import STOPWORDS, WordCloud
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from email.message import Message
import logging
from typing import List, Dict, Callable
import pickle
from networkx.exception import NetworkXNoCycle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from difflib import SequenceMatcher
import functools as ft
from sklearn import preprocessing
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from scipy import stats
from joblib import dump
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import wraps
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.util import ngrams
import yaml
from recordclass import recordclass
