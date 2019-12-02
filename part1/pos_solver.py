#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: Sharad Singh (singshar), Scott Steinbruegge (srsteinb), Bobby Rathore (brathore)
#
# (Based on skeleton code by D. Crandall)
#
####
"""
During initial training process, initial probability, transition probability and emission probability is calculated.
Simple: Here we have used the naive baye's approach to calculate a posterior given a prior and likelihood where denominator can be ingnored being a normalization 
constant. Taking log of posterior is to get a cost function and choosing the minimum cost is the idea
HMM using Viterbi: Here we are calculating  a chain of bayesian networks with their emission probability,where the emission probability of previous state is multiplied to current state and 
the sequence is repeated to observe a pattern and the sequence with MAP is considered .
Complex:Here initially all words are supposed to be noun and gibbs sampling is done for repition=6000 and warmup=2500. Each POS for gibbs sampling , transition probability
b/w current and previous state is multiplied with transition probability b/w current, state just before previous and emission prob of POS.
Assumptions:
There can be situations where the word is not available in the dictionary as it would not have occured hence it's emission probability is assumed to be 10^-10
 Word and sentency accuracy achieved with the bc.test dataset:
                      Words correct:     Sentences correct: 
      Ground truth:      100.00%              100.00%
            Simple:       97.62%               66.67%
               HMM:       97.84%               66.86%
           Complex:       92.48%               45.35%
"""

import math
import numpy as np
from collections import Counter

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.


class Solver:
    emission_probability = {}
    position_list = []
    initial_probability = {}
    second_probability = {}
    transition_probability = {}
    posterior_probability = {}
    other_probability = {}
    pos_sen_cnt = 1
    sen_cnt = 0
    Posterior_Model = {"Simple": {}, "Complex": {}, "HMM": {}}

    def posterior(self, model, sentence, label):
        """
        Given a model , sentence and POS , logarithmic value of posterior probability can be calculated
        """

        if model == "Simple":
            cost = sum(
                [
                    (
                        (math.log(self.emission_probability[label[i]][sentence[i]]))
                        + (math.log(self.posterior_probability[label[i]]))
                        if sentence[i] in self.emission_probability[label[i]]
                        else (math.log(1 / float(10 ** 10)))
                        + (math.log(self.posterior_probability[label[i]]))
                    )
                    for i in range(len(sentence))
                ]
            )
            return cost
        elif model == "Complex":
            post_array = []
            for i in range(len(sentence)):
                if i == 0:
                    post_array.append(
                        self.emission_probability[label[i]][sentence[i]]
                        * self.initial_probability[label[i]]
                        if sentence[i] in self.emission_probability[label[i]]
                        else (1 / float(10 ** 10)) * self.initial_probability[label[i]]
                    )
                elif i == 1:
                    post_array.append(
                        self.emission_probability[label[i]][sentence[i]]
                        * (
                            self.transition_probability[label[i - 1]][label[i]]
                            * self.posterior_probability[label[i - 1]]
                            / self.posterior_probability[label[i]]
                        )
                        * self.posterior_probability[label[i]]
                        if sentence[i] in self.emission_probability[label[i]]
                        else (1 / float(10 ** 10))
                        * (
                            self.transition_probability[label[i - 1]][label[i]]
                            * self.posterior_probability[label[i - 1]]
                            / self.posterior_probability[label[i]]
                        )
                        * self.posterior_probability[label[i]]
                    )
                else:
                    post_array.append(
                        self.emission_probability[label[i]][sentence[i]]
                        * (
                            self.transition_probability[label[i - 1]][label[i]]
                            * self.posterior_probability[label[i - 1]]
                            / self.posterior_probability[label[i]]
                        )
                        * (
                            self.transition_probability[label[i - 1]][label[i]]
                            * self.posterior_probability[label[i - 2]]
                            / self.posterior_probability[label[i]]
                        )
                        * self.posterior_probability[label[i]]
                        if sentence[i] in self.emission_probability[label[i]]
                        else (1 / float(10 ** 10))
                        * (
                            self.transition_probability[label[i - 1]][label[i]]
                            * self.posterior_probability[label[i - 1]]
                            / self.posterior_probability[label[i]]
                        )
                        * (
                            self.transition_probability[label[i - 2]][label[i]]
                            * self.posterior_probability[label[i - 2]]
                            / self.posterior_probability[label[i]]
                        )
                        * self.posterior_probability[label[i]]
                    )
            post_array = [math.log(p) for p in post_array]
            cost = sum(post_array)
            return cost

        elif model == "HMM":
            post_array = []
            for i in range(len(sentence)):
                if i == 0:
                    post_array.append(
                        (
                            self.initial_probability[label[i]]
                            * self.emission_probability[label[i]][sentence[i]]
                        )
                        if sentence[i] in self.emission_probability[label[i]]
                        else (self.initial_probability[label[i]] * (1 / float(10 ** 8)))
                    )
                else:
                    emi = (
                        (self.emission_probability[label[i]][sentence[i]])
                        if sentence[i] in self.emission_probability[label[i]]
                        else (1 / float(10 ** 10))
                    )

                    min_val = post_array[i - 1] * (
                        (self.transition_probability[label[i - 1]][label[i]])
                    )

                    post_array.append(emi * min_val)

            post_array = [math.log(p) for p in post_array]

            cost = sum(post_array)

            return cost
        else:
            print("Unknown algorithm!")

    def train(self, data):
        """
        Given a data, calculate initial,transition and emission probability
        """
        pos_list = [
            "adj",
            "adv",
            "adp",
            "conj",
            "det",
            "noun",
            "num",
            "pron",
            "prt",
            "verb",
            "x",
            ".",
        ]
        print("Train")

        wordpos_list = [
            tuple([line[0][i], line[1][i]])
            for line in data
            for i in range(len(line[0]))
        ]

        pos_dict = {pos: {} for pos in pos_list}

        wordpos_count = Counter(wordpos_list)

        for w in wordpos_count:
            pos_dict[w[1]].update({w[0]: wordpos_count[w]})
        posterior_prob = {}
        emission_prob = {}
        for pos in pos_dict.keys():
            posterior_prob[pos] = float(sum(pos_dict[pos].values())) / len(wordpos_list)
        for pos in pos_dict.keys():
            emission_prob[pos] = {
                word: float(pos_dict[pos][word]) / sum(pos_dict[pos].values())
                for word in pos_dict[pos].keys()
            }

        trans_count = {}
        transition_prob = {}
        for pos in pos_list:
            trans_count[pos] = {}
            transition_prob[pos] = {}
        pair_list = [
            tuple([line[1][i], line[1][i + 1]])
            for line in data
            for i in range(len(line[1]) - 1)
        ]

        unique_list = list(set(pair_list))

        for element in unique_list:
            trans_count[element[0]].update({element[1]: pair_list.count(element)})

        for pos in pos_list:
            transition_prob[pos] = {pos: (1 / float(10 ** 8)) for pos in pos_list}
            for key, value in trans_count[pos].items():
                transition_prob[pos].update(
                    {key: (value / float(sum(trans_count[pos].values())))}
                )
        initial_list = [line[1][0] for line in data]
        initial_count = Counter(initial_list)
        initial_prob = {
            pos: float(initial_count[pos]) / sum(initial_count.values())
            for pos in initial_count.keys()
        }
        self.position_list = pos_list
        self.emission_probability = emission_prob
        self.transition_probability = transition_prob
        self.initial_probability = initial_prob
        self.posterior_probability = posterior_prob

    def posterior_first(self, word):
        """
        Given a first word in a sentence , calculate the posterior for all POS
    
        """
        prob = {}
        if word not in prob.keys():
            prob[word] = {
                pos: self.emission_probability[pos][word]
                * self.initial_probability[pos]
                if word in self.emission_probability[pos]
                else (1 / float(10 ** 10)) * self.initial_probability[pos]
                for pos in self.position_list
            }

        return prob[word]

    def post_second(self, word, previous):
        """
        Given a second word in a sentence , calculate the posterior for all POS
        """
        prob = {}
        if word not in prob.keys():

            prob[word] = {
                pos: self.emission_probability[pos][word]
                * (
                    float(
                        self.transition_probability[previous][pos]
                        * self.posterior_probability[previous]
                    )
                    / self.posterior_probability[pos]
                )
                * self.posterior_probability[pos]
                if word in self.emission_probability[pos]
                else (1 / float(10 ** 10))
                * (
                    float(
                        self.transition_probability[previous][pos]
                        * self.posterior_probability[previous]
                    )
                    / self.posterior_probability[pos]
                )
                * self.posterior_probability[pos]
                for pos in self.position_list
            }

        return prob[word]

    def posterior_else(self, word, previous, previous_second):
        """
        This function calculated posterior probability of words other than
        first and second one and calculates posterior of all parts of speech.
        """
        prob = {}
        if word not in prob.keys():

            prob[word] = {
                pos: self.emission_probability[pos][word]
                * (
                    float(
                        self.transition_probability[previous][pos]
                        * self.posterior_probability[previous]
                    )
                    / self.posterior_probability[pos]
                )
                * (
                    float(
                        self.transition_probability[previous_second][pos]
                        * self.posterior_probability[previous_second]
                    )
                    / self.posterior_probability[pos]
                )
                * self.posterior_probability[pos]
                if word in self.emission_probability[pos]
                else (1 / float(10 ** 10))
                * (
                    float(
                        self.transition_probability[previous][pos]
                        * self.posterior_probability[previous]
                    )
                    / self.posterior_probability[pos]
                )
                * (
                    float(
                        self.transition_probability[previous_second][pos]
                        * self.posterior_probability[previous_second]
                    )
                    / self.posterior_probability[pos]
                )
                * self.posterior_probability[pos]
                for pos in self.position_list
            }

        return prob[word]

    def simple(self, sentence):
        """
        Given a sentence, list of POS labels are calculated using Naive Bayes
        """
        output_list = [
            self.position_list[
                np.argmax(
                    [
                        (math.log(self.emission_probability[pos][sentence[i]]))
                        + (math.log(self.posterior_probability[pos]))
                        if sentence[i] in self.emission_probability[pos]
                        else (math.log(1 / float(10 ** 10)))
                        + (math.log(self.posterior_probability[pos]))
                        for pos in self.position_list
                    ]
                )
            ]
            for i in range(len(sentence))
        ]
        return output_list

    def complex(self, sentence):
        """
        Given a sentence, list of POS labels are calculated using MCMC algo
        """
        repition = 6000
        warmup = 2500
        pos_mcmc_dict = {"pos_" + str(i): {} for i in range(len(sentence))}
        sequence = ["noun"] * len(sentence)
        for i in range(len(sentence)):
            if i == 0:
                prob_first = self.posterior_first(sentence[i])
                sample_first = list(
                    np.random.choice(
                        [keys for keys in prob_first.keys()],
                        repition,
                        p=[
                            float(prob_first[keys]) / sum(prob_first.values())
                            for keys in prob_first.keys()
                        ],
                    )
                )
                sample_first = sample_first[warmup:]
                pos_mcmc_dict["pos_" + str(i)] = {
                    pos: (float(sample_first.count(pos)) / len(sample_first))
                    for pos in self.position_list
                }
                sequence[i] = max(
                    pos_mcmc_dict["pos_" + str(i)],
                    key=pos_mcmc_dict["pos_" + str(i)].get,
                )
            elif i == 1:
                prob_second = self.post_second(sentence[i], sequence[i - 1])
                sample_second = list(
                    np.random.choice(
                        [keys for keys in prob_second.keys()],
                        repition,
                        p=[
                            float(prob_second[keys]) / sum(prob_second.values())
                            for keys in prob_second.keys()
                        ],
                    )
                )
                sample_second = sample_second[warmup:]
                pos_mcmc_dict["pos_" + str(i)] = {
                    pos: (float(sample_second.count(pos)) / len(sample_second))
                    for pos in self.position_list
                }
                sequence[i] = max(
                    pos_mcmc_dict["pos_" + str(i)],
                    key=pos_mcmc_dict["pos_" + str(i)].get,
                )
            else:
                prob_other = self.posterior_else(
                    sentence[i], sequence[i - 1], sequence[i - 2]
                )
                sample_other = list(
                    np.random.choice(
                        [keys for keys in prob_other.keys()],
                        repition,
                        p=[
                            float(prob_other[keys]) / sum(prob_other.values())
                            for keys in prob_other.keys()
                        ],
                    )
                )
                sample_other = sample_other[warmup:]
                pos_mcmc_dict["pos_" + str(i)] = {
                    pos: (float(sample_other.count(pos)) / len(sample_other))
                    for pos in self.position_list
                }
                sequence[i] = max(
                    pos_mcmc_dict["pos_" + str(i)],
                    key=pos_mcmc_dict["pos_" + str(i)].get,
                )
        return sequence

    def hmm(self, sentence):

        """
        Input is a sentence and returns POS label list using viterbi algorithm
        """
        viterbi_dict = {i: {} for i in range(len(sentence))}

        for i in range(len(sentence)):
            temp_dict = {}
            if i == 0:
                for pos in self.position_list:
                    temp_dict[pos] = tuple(
                        [
                            (
                                (
                                    (-math.log(self.initial_probability[pos]))
                                    + (
                                        -math.log(
                                            self.emission_probability[pos][sentence[i]]
                                        )
                                    )
                                )
                                if sentence[i] in self.emission_probability[pos]
                                else (
                                    (-math.log(self.initial_probability[pos]))
                                    + (-math.log((1 / float(10 ** 8))))
                                )
                            ),
                            "Start",
                        ]
                    )
                viterbi_dict[i] = temp_dict

            else:
                for pos in self.position_list:
                    emi = (
                        -math.log(self.emission_probability[pos][sentence[i]])
                        if sentence[i] in self.emission_probability[pos]
                        else (-math.log((1 / float(10 ** 8))))
                    )
                    min_val = min(
                        [
                            (
                                (
                                    viterbi_dict[i - 1][pos_prev][0]
                                    + (
                                        -math.log(
                                            self.transition_probability[pos_prev][pos]
                                        )
                                    )
                                ),
                                pos_prev,
                            )
                            for pos_prev in self.position_list
                        ]
                    )
                    temp_dict[pos] = tuple([emi + min_val[0], min_val[1]])
                viterbi_dict[i] = temp_dict
            i = i + 1
        pos_list = []
        prev = ""
        for i in range(len(sentence) - 1, -1, -1):

            if i == len(sentence) - 1:
                minimum = min(list(viterbi_dict[i].values()))

                for key in viterbi_dict[i].keys():
                    if (
                        viterbi_dict[i][key][0] == minimum[0]
                        and viterbi_dict[i][key][1] == minimum[1]
                    ):

                        pos_list.append(key)
                        prev = minimum[1]
            else:
                pos_list.append(prev)
                prev = viterbi_dict[i][prev][1]

        pos_list.reverse()

        return pos_list

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):

        """
        Given a model and sentence, labels are calculated
        """
        if model == "Simple":
            return self.simple(sentence)
        elif model == "Complex":
            return self.complex(sentence)
        elif model == "HMM":
            return self.hmm(sentence)
        else:
            print("Unknown algorithm!")
