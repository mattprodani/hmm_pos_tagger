import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import pickle

class MaxEntropyModel:
    """
        Logistic Regression POS Tagger
        File input: <word>\t<tag> (one word per line) separate sentences with empty line

        @author: mp5908 - Matt Prodani - NYU
    
    """
    def __init__(self, train_files) -> None:
        self.PREFIX_MAX = 4
        self.ENG_FEATURES = 6
        self.train_files = train_files if type(train_files) == list else [train_files]
        self.vocab = self.get_vocab(self.train_files)
        self.prefix_dict, self.suffix_dict = self.create_prefixes(self.vocab)
        self.FEATURE_SIZE = self.ENG_FEATURES + len(self.prefix_dict) + len(self.suffix_dict)

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.make_dataset(self.train_files)

    def train(self, num_iter=50):
        self.model = LogisticRegression(max_iter=num_iter, multi_class='multinomial', solver='lbfgs', verbose = 1, n_jobs=-1)
        self.model.fit(self.X_train, self.Y_train)
    
    def evaluate(self):
        self.model.predict_proba(self.X_test)
        print(f"Classifier Accuracy on Test Set: {self.model.score(self.X_test, self.Y_test)}")
        
    def clear_train_data(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
    
    def get_emission_vector(self, word, tag_to_idx):
        features = self.extract_features(word)
        placeholder = np.zeros(len(tag_to_idx))
        placeholder.fill(-np.inf)
        output = self.model.predict_log_proba(features.reshape(1, -1))
        for class_, prob in zip(self.model.classes_, output[0]):
            placeholder[tag_to_idx[class_]] = prob
        return placeholder


    def make_dataset(self, file):
        dfs = [pd.read_table(file, sep='\t', header=None, names=['word', 'tag']).dropna() for file in self.train_files]
        dataset = pd.concat(dfs)

        dataset.drop_duplicates(inplace=True)
        dataset['tag'] = dataset['tag'].str.upper()

        X = self.batch_extract(dataset['word'])
        Y = dataset['tag'].to_list()
        print(f"dataset size: {len(dataset)}")
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        return X_train, Y_train, X_test, Y_test

    def batch_extract(self, words):
        X = np.zeros((len(words), self.FEATURE_SIZE))
        for i, word in enumerate(words):
            X[i] = self.extract_features(word)
        return X
            
    def extract_features(self, word):
        features = np.zeros(self.FEATURE_SIZE)
        features[0] = np.any([c.isdigit() for c in word])
        features[1] = np.any([c.isupper() for c in word])
        features[2] = '-' in word
        features[3] = word.isupper()
        features[4] = features[0] and features[1] and features[2]
        for i in range(1, max(len(word), self.PREFIX_MAX)):
            if word[:i] in self.prefix_dict:
                features[self.prefix_dict[word[:i]]] = 1
            if word[-i:] in self.suffix_dict:
                features[self.suffix_dict[word[-i:]]] = 1
        return features



    def get_vocab(self, train_files):
        vocab = set()
        for file in train_files:
            with open(file, 'r') as f:
                for line in f:
                    if line == "\n": continue
                    word = line.split('\t')[0]
                    vocab.add(word)
        return vocab

    def create_prefixes(self, vocab):
        prefixes = defaultdict(int)
        suffixes = defaultdict(int)
        for word in vocab:
            for i in range(1, self.PREFIX_MAX):
                prefixes[word[:i]] += 1
                suffixes[word[-i:]] += 1
        # remove rare prefixes
        prefixes = {k: v for k, v in prefixes.items() if v > 5}
        suffixes = {k: v for k, v in suffixes.items() if v > 5}

        prefix_dict = {k: i for i, k in enumerate(prefixes.keys(), start = self.ENG_FEATURES)}
        suffix_dict = {k: i for i, k in enumerate(suffixes.keys(), start = self.ENG_FEATURES + len(prefixes))}

        print(f"prefixes: {len(prefixes)}")
        print(f"suffixes: {len(suffixes)}")
        return prefix_dict, suffix_dict


class MaxEntropyLookModel:
    """
        POC for look-ahead model. Haven't updated HHM to support this as it 
        doesn't seem to improve overall performance for how long it takes to train.
    
    """
    def __init__(self, train_files) -> None:
        self.PREFIX_MAX = 4
        self.ENG_FEATURES = 6
        self.train_files = train_files if type(train_files) == list else [train_files]
        self.vocab = self.get_vocab(self.train_files)
        self.prefix_dict, self.suffix_dict = self.create_prefixes(self.vocab)
        self.FEATURE_SIZE = self.ENG_FEATURES + len(self.prefix_dict) + len(self.suffix_dict)

        self.X_train, self.Y_train, self.X_test, self.Y_test = self.make_dataset(self.train_files)

    def train(self, num_iter=50):
        self.model = LogisticRegression(max_iter=num_iter, multi_class='multinomial', solver='lbfgs', verbose = 1, n_jobs=-1)
        self.model.fit(self.X_train, self.Y_train)
    
    def evaluate(self):
        self.model.predict_proba(self.X_test)
        print(f"Classifier Accuracy on Test Set: {self.model.score(self.X_test, self.Y_test)}")
        
    def clear_train_data(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)
    
    def get_emission_vector(self, word1, word2, tag_to_idx):
        features = self.look_vector(word1, word2)
        placeholder = np.zeros(len(tag_to_idx))
        placeholder.fill(-np.inf)
        output = self.model.predict_log_proba(features.reshape(1, -1))
        for class_, prob in zip(self.model.classes_, output[0]):
            placeholder[tag_to_idx[class_]] = prob
        return placeholder


    def make_dataset(self, file):
        samples = set()
        
        for file in self.train_files:
            with open(file) as f:
                prev_word = "<S>"
                for line in f:
                    if line == '\n':
                        prev_word = "<S>"
                    else:
                        word, pos = line.strip().split()
                        pos = pos.upper()
                        samples.add((prev_word, word, pos))
                        prev_word = word
        

        X, Y = self.batch_extract(list(samples))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
        return X_train, Y_train, X_test, Y_test

    def batch_extract(self, samples):
        X = np.zeros((len(samples), self.FEATURE_SIZE * 2))
        Y = []
        cache = {}
        for i, (prev_word, word, pos) in enumerate(samples):
            if prev_word not in cache:
                cache[prev_word] = self.extract_features(prev_word)
            if word not in cache:
                cache[word] = self.extract_features(word)
            X[i, :self.FEATURE_SIZE] = cache[word]
            X[i, self.FEATURE_SIZE:] = cache[prev_word]
            Y.append(pos)
        return X, Y
            
    def look_vector(self, prev_s, curr_s):
        samples = [(prev_s, curr_s)]
        X = np.zeros((len(samples), self.FEATURE_SIZE * 2))
        cache = {}
        for i, (prev_word, word, pos) in enumerate(samples):
            if prev_word not in cache:
                cache[prev_word] = self.extract_features(prev_word)
            if word not in cache:
                cache[word] = self.extract_features(word)
            X[i, :self.FEATURE_SIZE] = cache[word]
            X[i, self.FEATURE_SIZE:] = cache[prev_word]
        return X
    
    def extract_features(self, word):
        features = np.zeros(self.FEATURE_SIZE)
        if word == '<S>':
            features[5] = 1
            return features
        features[0] = np.any([c.isdigit() for c in word])
        features[1] = np.any([c.isupper() for c in word])
        features[2] = '-' in word
        features[3] = word.isupper()
        features[4] = features[0] and features[1] and features[2]
        for i in range(1, max(len(word), self.PREFIX_MAX)):
            if word[:i] in self.prefix_dict:
                features[self.prefix_dict[word[:i]]] = 1
            if word[-i:] in self.suffix_dict:
                features[self.suffix_dict[word[-i:]]] = 1
        return features



    def get_vocab(self, train_files):
        vocab = set(("<S>"))
        for file in train_files:
            with open(file, 'r') as f:
                for line in f:
                    if line == "\n": continue
                    word = line.split('\t')[0]
                    vocab.add(word)
        return vocab

    def create_prefixes(self, vocab):
        prefixes = defaultdict(int)
        suffixes = defaultdict(int)
        for word in vocab:
            for i in range(1, self.PREFIX_MAX):
                prefixes[word[:i]] += 1
                suffixes[word[-i:]] += 1
        # remove rare prefixes
        prefixes = {k: v for k, v in prefixes.items() if v > 5}
        suffixes = {k: v for k, v in suffixes.items() if v > 5}

        prefix_dict = {k: i for i, k in enumerate(prefixes.keys(), start = self.ENG_FEATURES)}
        suffix_dict = {k: i for i, k in enumerate(suffixes.keys(), start = self.ENG_FEATURES + len(prefixes))}

        print(f"prefixes: {len(prefixes)}")
        print(f"suffixes: {len(suffixes)}")
        return prefix_dict, suffix_dict


