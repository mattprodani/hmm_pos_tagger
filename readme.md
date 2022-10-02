# Hidden-Markov-Model Parts of Speech Classifier
## Mathias Prodani - Natural Language Processing - CSCI 480 
------------

Overview
-----------
To begin, the model is based on the bigram Viterbi algorithm with a Logistic Regression (MaxEnt) 
model to handle unseen words. This is used to predict the most likely sequence of POS in a sentence. 
To train, I tokenized the words and maintained dictionaries so I could use NumPy matrices, 
making the calculations easier. The words are then processed and using the Transition and Emission 
matrix, the model can find state probabilities for new content.

Methodology and optimizations
---------------------
- Pre-tokenized words

- Logarithmic probabilities
Instead of using standard n/N probabilities, this program calculates in log space, with 0 being -inf.
This allows for faster calculations, since multiplications in log-space become addition, and prevents underflow
when probability products get too low.

- Laplace Smoothing
To "smooth out" rarely occuring words, the model uses Laplace smoothing, and an alpha parameter can be provided
at train time which will affect the impact of the smoothing. For this case, I found that an alpha of 0.001 performs 
best.

Handling OOV words
-------------------
- Maximum Entropy Classification Approach 
I wanted to categorize words based on lexical properties. Finding and hardcoding the correct probabilities 
proved quite difficult though, so I landed on Logistic Regression to create a model to predict emission 
probabilities for unseen classes. 
Based on Ratnaparkhi (1996) I used the following categories to create boolean feature vectors:

* Contains digit
* Contains uppercase letter
* Contains hyphen
* Is UPPERCASE 
* Contains digit, uppercase, and hyphen
* Has common prefix up to 4-letters
    To accomplish this, I found all the 1-4 letter prefixes that
    occur more than 5 times in the training data, and each one of them
    is a feature in the vector. If a word starts with aba, v(word)["aba"] = 1,
    assuming "aba" occurs often as a prefix
* Has common suffix up to 4-letters

The model was able to achieve 67% accuracy on predicting the POS of a word solely from these variables, which
makes it extremely effective when predicting class probabilities.
Ratnaparkhi also uses surrounding words' feature vectors for prediction, but I did not implement or test that.

After the model is trained, logarithmic class probabilities are calculated for unkown words and used as emission 
vectors for Viterbi.

This brought the Viterbi accuracy from 90% to 96% on the test set.

Requirements 
----------
Tested on Python3.8

scikit-learn
pandas
NumPy
tqdm (for progress bars)

To install: "pip install scikit-learn pandas numpy tqdm"

Usage
-----------
classifier.py and hmm.py contain the classes for the feature vector MaxEnt model and HMM model.
run.py is a convenience script that will go through the training pipeline.

To train/test run "python run.py" and follow the prompts. You can provide a pickled file and have the option to save the classifier,
but training takes less than a minute on my machine, so I don't assume it is an issue. The script provides default config 
values for alpha, and MaxEnt iterations which can be customized. It also allows providing multiple input files.

References:
-------------
Ratnaparkhi, Adwait. 1996. A maximum entropy model for part-of-speech tagging. In In Proceedings of the Conference on Empirical Methods in Natural
Language Processing, pages 133–142.

Jurafsky, Daniel and James H. Martin. 2008. Speech and Language Processing,
An Introduction to Natural Language Processing,Computational Linguistics,
and Speech Recognition. Prentice-Hall, 2nd edition.