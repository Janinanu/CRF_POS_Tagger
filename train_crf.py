import argparse
import re
import pickle
import math
from collections import defaultdict
import random
import operator

class Trainer:

    def __init__(self, train_file):

        self.train_data = []
        self.tagset = set()

        self.cache_shapes = defaultdict(str)
        self.cache_substrings = defaultdict(list)

        self.weights = defaultdict(float)

        with open(train_file) as f:
            self.read_data(f)

        #double check
        forward, cache_lex, cache_context = self.forward_scores(["<s>", "The", "Computer", "knows", "me", "<s>"])
        backward = self.backward_scores(["<s>", "The", "Computer", "knows", "me", "<s>"], forward, cache_lex, \
                                        cache_context)
        print(len(forward))
        for d in forward:
            print(d)
        print("###")
        print(len(backward))
        for d in backward:
            print(d)
        print(forward[-1]["<s>"] == backward[0]["<s>"])

        self.train(0.01)

    def read_data(self, file):

        sentence_words = []
        sentence_tags = []
        boundary_symbol = ["<s>"]

        for line in file:
            if line != "\n":
                word, tag = line.strip().split("\t")
                sentence_words.append(word)
                sentence_tags.append(tag)
                self.tagset.add(tag)
            else:
                self.train_data.append((boundary_symbol + sentence_words + boundary_symbol, \
                                        boundary_symbol + sentence_tags + boundary_symbol))
                sentence_words = []
                sentence_tags = []

    def shape(self, word):

            digits = re.sub("\d", "9", word)
            capital = re.sub("[A-Z]", "X", digits)
            lower = re.sub("[a-z]", "x", capital)
            merge = re.sub(r"(.)\1{1,}", r"\1", lower)

            return merge

    def add_shapes(self, word):

        shape = self.shape(word)
        self.cache_shapes[word] = shape

    def add_substrings(self, padded):

        for window in range(2, 7):
            for char_pos in range(len(padded) - window + 1):
                substr = padded[char_pos:char_pos + window]
                self.cache_substrings[padded].append(substr)

    def get_lex_features(self, tag, words, i):

            TW_t_w =  "_".join(["TW", tag, words[i]])
            TL_t_lower = "_".join(["TL", tag, words[i].lower()])

            if words[i] not in self.cache_shapes:
                self.add_shapes(words[i])
            TF_t_shape = "_".join(["TF", tag, self.cache_shapes[words[i]]])

            if i+1 < len(words):
                next_word = words[i+1]
            else:
                next_word = "<s>"
            TWW_word_next = "_".join(["TWW+1", tag, words[i], next_word])

            if words[i-1]:
                prev_word = words[i-1]
            else:
                prev_word = "<s>"
            TWminus1W = "_".join(["TW-1W", tag, prev_word, words[i]])

            TC = "_".join(["TC", tag, words[i].title()])

            lex_feature_list = [TW_t_w, TL_t_lower, TF_t_shape, TWW_word_next, TWminus1W, TC]

            padded = "#" + words[i] + "#"
            if padded not in self.cache_substrings:
                self.add_substrings(padded)
            for substr in self.cache_substrings[padded]:
                lex_feature_list.append("_".join(["TSUBS", tag, substr]))

            return lex_feature_list

    def get_context_features(self, prev_tag, tag, words, i):

            TT = "_".join(["TT", prev_tag, tag])
            TTW = "_".join(["TTW", prev_tag, tag, words[i]])
            TTWminus1 = "_".join(["TTW-1", prev_tag, tag, words[i-1]])

            return [TT, TTW, TTWminus1]

    def multiply_weights(self, feature_list):

        dot_product = 0.0
        for feature in feature_list:
                dot_product += self.weights[feature]

        return dot_product

    def log_add(self, a, b):

        return a + math.log(1 + math.exp(b-a)) if a >= b else b + math.log(1 + math.exp(a-b))

    def forward_scores(self, words):

        forward = [{"<s>": 0}]

        cache_lex = defaultdict(tuple)
        cache_context = defaultdict(tuple)

        for i in range(1, len(words)):

            forward.append(defaultdict(float))

            if i == len(words) - 1:

                tag = words[i]

                lex_feature_list = self.get_lex_features(tag, words, i)
                lex_score = self.multiply_weights(lex_feature_list)

                cache_lex[(tag, words[i])] = (lex_feature_list, lex_score)

                for prev_tag in forward[i-1]:

                    context_feature_list = self.get_context_features(prev_tag, tag, words, i)
                    context_score = math.exp(self.multiply_weights(context_feature_list))

                    cache_context[(prev_tag, tag, words[i], words[i-1])] = (context_feature_list, context_score)

                    score = lex_score + context_score + forward[i-1][prev_tag]

                    if tag in forward[i]:
                        forward[i][tag] = self.log_add(score, forward[i][tag])
                    else:
                        forward[i][tag] = self.log_add(score, -float("inf"))
            else:

                for tag in self.tagset:

                    lex_feature_list = self.get_lex_features(tag, words, i)
                    lex_score = self.multiply_weights(lex_feature_list)

                    cache_lex[(tag, words[i])] = (lex_feature_list, lex_score)

                    for prev_tag in forward[i-1]:

                        context_feature_list = self.get_context_features(prev_tag, tag, words, i)
                        context_score = math.exp(self.multiply_weights(context_feature_list))

                        cache_context[(prev_tag, tag, words[i], words[i-1])] = (context_feature_list, context_score)

                        score = lex_score + context_score + forward[i-1][prev_tag]

                        if tag in forward[i]:
                            forward[i][tag] = self.log_add(score, forward[i][tag])
                        else:
                            forward[i][tag] = self.log_add(score, -float("inf"))

            best_tags = [tag for tag, _ in sorted(forward[i].items(), key=operator.itemgetter(1), reverse=True)[:4]]
            new_tag_dict = defaultdict(float)

            for tag, value in forward[i].items():
                if tag in best_tags:
                    new_tag_dict[tag] = value
            forward[i] = new_tag_dict

        return forward, cache_lex, cache_context

    def backward_scores(self, words, forward, cache_lex, cache_context):

        backward = [defaultdict(float) for i in range(len(words) - 1)] + [{"<s>": 0}]  # Länge 5, bis Index 4

        for i in reversed(range(len(words)-1)):  # ab Index 3 bis 0

            if i == 0:

                prev_tag = words[i]

                if (prev_tag, words[i]) in cache_lex:
                    lex_feature_list, lex_score = cache_lex[(prev_tag, words[i])]
                else:
                    lex_feature_list = self.get_lex_features(prev_tag, words, i)
                    lex_score = self.multiply_weights(lex_feature_list)

                for tag in backward[i+1]:

                    if (prev_tag, tag, words[i], words[i-1]) in cache_context:
                        context_feature_list, context_score = cache_context[(prev_tag, tag, words[i], words[i-1])]
                    else:
                        context_feature_list = self.get_context_features(prev_tag, tag, words, i+1)
                        context_score = math.exp(self.multiply_weights(context_feature_list))

                    score = lex_score + context_score + backward[i+1][tag]

                    if prev_tag in backward[i]:
                        backward[i][prev_tag] = self.log_add(score, backward[i][prev_tag])
                    else:
                        backward[i][prev_tag] = self.log_add(score, -float("inf"))

            else:

                for tag in self.tagset:

                    if tag in forward[i]:

                        if (tag, words[i]) in cache_lex:
                            lex_feature_list, lex_score = cache_lex[(tag, words[i])]
                        else:
                            lex_feature_list = self.get_lex_features(tag, words, i)
                            lex_score = self.multiply_weights(lex_feature_list)

                        for next_tag in backward[i+1]:

                            if (tag, next_tag, words[i+1], words[i]) in cache_context:
                                context_feature_list, context_score = cache_context[(tag, next_tag, \
                                                                                     words[i+1], words[i])]
                            else:
                                context_feature_list = self.get_context_features(next_tag, tag, words, i)
                                context_score = math.exp(self.multiply_weights(context_feature_list))

                            score = lex_score + context_score + backward[i+1][next_tag]

                            if tag in backward[i]:
                                backward[i][tag] = self.log_add(score, backward[i][tag])
                            else:
                                backward[i][tag] = self.log_add(score, -float("inf"))

        return backward


    def expected_feature_counts(self, words):

        exp_count = defaultdict(float)

        forward, cache_lex, cache_context = self.forward_scores(words)
        backward = self.backward_scores(words, forward, cache_lex, cache_context)

        for i in range(1, len(words)):

            tag_scores_forward = defaultdict(float)

            for tag in forward[i]:

                if (tag, words[i]) in cache_lex:
                    lex_feature_list, lex_score = cache_lex[(tag, words[i])]
                else:
                    lex_feature_list = self.get_lex_features(tag, words, i)
                    lex_score = self.multiply_weights(lex_feature_list)

                tag_scores_forward[tag] = lex_score

            best_tags_forward = [tag for tag, score in
                             sorted(tag_scores_forward.items(), key=operator.itemgetter(1), reverse=True)[:8]]

            for tag in best_tags_forward:

                for prev_tag in forward[i-1]:

                        if (prev_tag, tag, words[i], words[i-1]) in cache_context:
                            context_feature_list, context_score = cache_context[(prev_tag, tag, words[i], words[i-1])]
                        else:
                            context_feature_list = self.get_context_features(prev_tag, tag, words, i)
                            context_score = math.exp(self.multiply_weights(context_feature_list))

                        p_prev_tag_tag = math.exp(forward[i-1][prev_tag] + lex_score + context_score + \
                                                  backward[i][tag] - forward[-1]["<s>"])

                        for feature in lex_feature_list + context_feature_list:
                            exp_count[feature] += p_prev_tag_tag

        return exp_count

    def observed_feature_counts(self, words, tags):

        observed_count = defaultdict(float)

        for i in range(len(words)):

                tag = tags[i]
                lex_feature_list = self.get_lex_features(tag, words, i)

                prev_tag = tags[i-1]
                context_feature_list = self.get_context_features(prev_tag, tag, words, i)

                for feature in lex_feature_list + context_feature_list:
                    observed_count[feature] += 1

        return observed_count

    def save_parameters(self, filename):

        with open(filename, "wb") as out:
            pickle.dump((self.weights, self.tagset), out)

    def train(self, learn_rate):

        for it in range(1000):

            print(it)
            #for feature, weight in self.weights.items():
               #if weight != 0.0:
                       #print(weight)

            words, tags = random.choice(self.train_data)

            exp_count = self.expected_feature_counts(words)
            observed_count = self.observed_feature_counts(words, tags)

            for feature in observed_count:
                self.weights[feature] += learn_rate*(observed_count[feature] - exp_count[feature])

        self.save_parameters(args.out_file)




parser = argparse.ArgumentParser(
    description="Train and evaluate CRF for tagging model")
parser.add_argument(
    "train_file", type=str, help="training sentences with tags")
parser.add_argument(
    "out_file", type=str, help="filename to store weights and tagset")
args = parser.parse_args()

crf = Trainer(args.train_file)
