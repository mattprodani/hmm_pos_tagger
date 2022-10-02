import numpy as np
from classifier_MP5908_HW3 import MaxEntropyModel
import mmap
from tqdm import tqdm



class POSTagger:
    """
    Viterbi algorithm for POS tagging
    Uses laplace smoothing and MaxEnt classifier

    Training file input: <word>\t<tag> (one word per line) separate sentences with empty line
    Inference file input: <word> (one word per line) separate sentences with empty line
    Output file format: <word>\t<tag> (one word per line) separate sentences with empty line
    @author: mp5908 - Matt Prodani - NYU

    """
    def __init__(self, train_files):
        self.train_files = train_files if type(train_files) == list else [train_files]

    
    def viterbi(self, TRANSITION, EMISSION, sequence):
        V = np.zeros((len(TRANSITION), len(sequence)))
        B = np.zeros((len(TRANSITION), len(sequence)), dtype=int)

        # initialization
        if isinstance(sequence[0], str): # if word is not in vocab
            emission_prob = self.classifier.get_emission_vector(sequence[0], self.tag_to_idx)
        else: emission_prob = EMISSION[:, sequence[0]]
        V[:, 0] = TRANSITION[0, :] + emission_prob

        # DP
        for w in range(1, len(sequence)):
            if isinstance(sequence[w], str): # if word is not in vocab
                emission_prob = self.classifier.get_emission_vector(sequence[w], self.tag_to_idx)
            else: emission_prob = EMISSION[:, sequence[w]]

            for q in range(len(TRANSITION)):
                V[q, w] = np.max(V[:, w-1] + TRANSITION[:, q] + emission_prob[q])
                B[q, w] = np.argmax(V[:, w-1] + TRANSITION[:, q])
        
        V[-1, -1] = np.max(V[:, -1] + TRANSITION[:, -1])

        # backtracking
        last = B[-1, -1] = np.argmax(V[:, -1] + TRANSITION[:, -1])
        tags = [last]
        for w in range(len(sequence)-1, 0, -1):
            last = B[int(last), w]
            tags.append(last)
        return tags[::-1]
        
        
    def train(self, classifier: MaxEntropyModel, alpha = 0.1):
        self.alpha = alpha
        self.classifier = classifier
        self.idx_to_token, self.idx_to_tag = self.create_vocab()
        self.tag_to_idx = {v: k for k, v in self.idx_to_tag.items()}
        self.token_to_idx = {v: k for k, v in self.idx_to_token.items()}
        raw_transitions, raw_emissions = self.create_transition_matrix( self.token_to_idx, self.tag_to_idx)
        self.TRANSITION, self.EMISSION = self.get_probabilities(raw_transitions, raw_emissions)

    def laplace_smoothing(self, TRANSITION, EMISSION):
        TRANSITION = (TRANSITION + self.alpha) / (TRANSITION.sum(axis=1, keepdims=True) + self.alpha*len(TRANSITION))
        EMISSION = (EMISSION + self.alpha) / (EMISSION.sum(axis=1, keepdims=True) + self.alpha*EMISSION.shape[1])
        return TRANSITION, EMISSION


    def get_probabilities(self, TRANSITION, EMISSION):

        # laplace smoothing
        if self.alpha > 0:
            TRANSITION, EMISSION = self.laplace_smoothing(TRANSITION, EMISSION)
        else:
            TRANSITION = TRANSITION / TRANSITION.sum(axis=1, keepdims=True)
            EMISSION = EMISSION / EMISSION.sum(axis=1, keepdims=True)

        # log probabilities
        with np.errstate(divide='ignore'):
            TRANSITION = np.log(TRANSITION / TRANSITION.sum(axis=1, keepdims=True))
            EMISSION = np.log(EMISSION / EMISSION.sum(axis=1, keepdims=True))

        # fix zero probabilities
        TRANSITION[np.isnan(TRANSITION)] = -np.inf
        EMISSION[np.isnan(EMISSION)] = -np.inf
        return TRANSITION, EMISSION



    def tokenize_sentence(self, sentence, token_to_idx):
        return [token_to_idx[word] if word in token_to_idx else word for word in sentence]

    def extract_tags(self, tokens, idx_to_tag, sentence):
        return [idx_to_tag[token] for token in tokens]

    def format_sentence(self, sentence, tags):
        output = ""
        for word, tag in zip(sentence, tags):
            output += f"{word}\t{tag}\n"
        return f"{output}\n"

    def create_vocab(self):
        tokens, pos_set = set(), set()
        for file in self.train_files:
            with open(file) as f:
                for line in f:
                    if line.strip() == '':
                        continue
                    else:
                        word, pos = line.strip().split()
                        tokens.add(word)
                        pos_set.add(pos.upper())
        pos_set = ["<S>"] + sorted(list(pos_set)) + ["</S>"]
        return dict(enumerate(tokens)), dict(enumerate(pos_set))


    def create_transition_matrix(self, token_to_idx, pos_to_idx):
        TRANSITION = np.zeros((len(pos_to_idx), len(pos_to_idx)))
        EMISSION = np.zeros((len(pos_to_idx), len(token_to_idx)))
        EMISSION[:, 0] = 1
        for file in self.train_files:
            with open(file) as f:
                prev_tag = 0
                for line in f:
                    if line == '\n':
                        TRANSITION[prev_tag, -1] += 1
                        prev_tag = 0
                    else:
                        word, pos = line.strip().split()
                        pos = pos.upper()
                        token, tag = token_to_idx[word], pos_to_idx[pos]
                        TRANSITION[prev_tag, tag] += 1
                        EMISSION[tag, token] += 1
                        prev_tag = tag
        return TRANSITION, EMISSION

    def predict(self, file, output_file):
        output = open(output_file, "w")
        with open(file) as f:
            sentence = []
            for line in tqdm(f, total=_get_num_lines(file)):
                if line == '\n':
                    pred = self.viterbi(self.TRANSITION, self.EMISSION, self.tokenize_sentence(sentence, self.token_to_idx))
                    pred_tags = self.extract_tags(pred, self.idx_to_tag, sentence)
                    output.write(self.format_sentence(sentence, pred_tags))
                    sentence = []
                else:
                    word = line.strip()
                    sentence.append(word)
        output.close()





def _get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
