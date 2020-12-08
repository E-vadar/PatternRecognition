import nltk
from functools import lru_cache

class Preprocessor:

    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stermmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.

        # The settings of stemmer, including 3 kinds, PorterStemmer, LancasterStemmer and SnowballStemmer('english')

        self.stem = lru_cache(maxsize=10000)(nltk.PorterStemmer().stem)
        # self.stem = lru_cache(maxsize=10000)(nltk.LancasterStemmer().stem)
        # self.stem = lru_cache(maxsize=10000)(nltk.SnowballStemmer('english').stem)

        # The settings of tokenizer, including 4 kinds, WhitespaceTokenizer, RegexpTokenizer('\w+'),
        # WordPunctTokenizer and NLTKWordTokenizer and SnowballStemmer('english')

        # self.tokenize = nltk.tokenize.WhitespaceTokenizer().tokenize
        self.tokenize = nltk.tokenize.RegexpTokenizer('\w+').tokenize
        # self.tokenize = nltk.tokenize.WordPunctTokenizer().tokenize
        # self.tokenize = nltk.tokenize.NLTKWordTokenizer().tokenize


    def __call__(self, text):

        # This is the usage of different settings of tokenizer

        # tokens = nltk.WhitespaceTokenizer().tokenize(text)
        tokens = nltk.RegexpTokenizer('\w+').tokenize(text)
        # tokens = nltk.WordPunctTokenizer().tokenize(text)
        # tokens = nltk.NLTKWordTokenizer().tokenize(text)

        tokens = [self.stem(token) for token in tokens]

        # This is the usage of lemmatizer

        # tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]

        return tokens
