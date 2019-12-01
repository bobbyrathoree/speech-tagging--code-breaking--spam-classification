#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: PLEASE PUT YOUR NAMES AND USERIDS HERE
#
# based on skeleton code by D. Crandall, 11/2019
#
# ./break_code.py : attack encryption
#

from __future__ import division
import random
import math
import copy
import sys
import threading
from collections import defaultdict
from itertools import cycle
from string import ascii_lowercase
from encode import encode_file, read_clean_file
from apply_code import get_initial_encryption_tables
import numpy as np
import time


decoded = False
all_characters = ascii_lowercase + " "


def animate(target: str, source: str):
    """
    Little helper function to display a loading screen.
    Inspired from: https://stackoverflow.com/a/22029635
    :param target: encoded file name
    :param source: source file name
    """
    for animation_instance in cycle(
        [
            " [=     ]",
            " [ =    ]",
            " [  =   ]",
            " [   =  ]",
            " [    = ]",
            " [     =]",
            " [    = ]",
            " [   =  ]",
            " [  =   ]",
            " [ =    ]",
        ]
    ):
        if decoded:
            break
        sys.stdout.write(
            "\rDecoding {0} using {1} ".format(target, source) + animation_instance
        )
        sys.stdout.flush()
        time.sleep(0.1)


def get_transitional_probabilities(corpus_text: str) -> defaultdict:
    """
    Function to compute the probabilities to transition from one letter to another
    :param corpus_text: source file text
    :return: transitional probabilities hash
    """
    transition_probabilities = defaultdict(
        dict, **{char: {} for char in all_characters}
    )

    # For each letter, first set all transitions to 1 (uniform)
    for from_letter in all_characters:
        for to_letter in all_characters:
            transition_probabilities[from_letter][to_letter] = 1

    # Increment transitions as per frequencies
    for index in range(1, len(corpus_text)):
        transition_probabilities[corpus_text[index - 1]][corpus_text[index]] += 1

    # Normalize them transition probabilities by dividing with their totals
    for _, value_dict in transition_probabilities.items():
        total = sum(value_dict.values())
        for key in value_dict.keys():
            value_dict[key] = value_dict[key] / total

    return transition_probabilities


def get_initial_probabilities(corpus_text: str) -> defaultdict:
    """
    Function to compute initial probabilities using the first letter of every word
    :param corpus_text: source file text
    :return: initial probabilities hash
    """
    initial_probabilities = defaultdict(float, **{char: 0 for char in all_characters})

    for word in corpus_text.split():
        initial_probabilities[word[0]] += 1

    total = sum(initial_probabilities.values())
    for key in initial_probabilities.keys():
        initial_probabilities[key] = initial_probabilities[key] / total

    return initial_probabilities


def get_probability_log(
    document_text: list,
    transition_probabilities: defaultdict,
    initial_probabilities: defaultdict,
) -> float:
    """
    Function to calculate log probability of document
    :param document_text: list of encoded words
    :param transition_probabilities: hash of transitional probabilities
    :param initial_probabilities: hash of initial probabilities
    :return: log probability of document
    """
    probability_of_entire_document = 0

    # Calculate the probability for every word
    for current_word in document_text:

        # If an empty list element, pass
        if len(current_word) < 1:
            continue

        # If only one letter word, get its initial probability log,
        # Else use and add relative transitional probabilities
        if len(current_word) == 1:
            probability_of_current_word = math.log(initial_probabilities[current_word[0]])
        else:
            probability_of_current_word = math.log(initial_probabilities[current_word[0]])
            for i in range(1, len(current_word)):
                if (transition_probabilities[current_word[i - 1]][current_word[i]]) > 0:
                    probability_of_current_word += math.log(
                        transition_probabilities[current_word[i - 1]][current_word[i]]
                    )

        # Add the log probability of each word to the probability of whole document
        probability_of_entire_document += probability_of_current_word

    return probability_of_entire_document


def modify_encryption_table(table: list, letters: bool = False) -> list:
    """
    Function to modify the encryption table (replace or rearrange)
    by swapping random letters or indices
    :param table: encryption table
    :param letters: flag as to whether to modify letters or indices
    :return: the new modified table
    """
    choices = list(ascii_lowercase) if letters else list(range(0, 4))
    first_choice = random.choice(choices)
    second_choice = random.choice(choices)

    # Make a new encryption table
    new_table = copy.deepcopy(table)

    # Swap using above choices
    new_table[first_choice], new_table[second_choice] = (
        new_table[second_choice],
        new_table[first_choice],
    )

    return new_table


# This function implements metropolis hasting algorithm to give proper decrypted document (hopefully!!!)
def break_code(encoded_text, corpus_text):
    """
    Function to showcase the Metropolis-Hastings algorithm to decrypt the encoded document.
    Referred from: https://bit.ly/2Y4D5qW
    :param encoded_text: encoded file contents
    :param corpus_text: the corpus file contents
    :return:
    """
    # Get the initial and transitional probabilities
    initial_probabilities = get_initial_probabilities(corpus_text)
    transition_probabilities = get_transitional_probabilities(corpus_text)

    # Number of iterations to run the sampling for
    epochs = 20000

    # Get the initial encryption (replacement and rearrangement) tables
    old_replace_table, old_rearrange_table = get_initial_encryption_tables()

    # Make a guess
    original_guess = encode_file(encoded_text, old_replace_table, old_rearrange_table)
    probability_of_original_guess = get_probability_log(
        original_guess.split(" "), transition_probabilities, initial_probabilities
    )

    updated_replacement_table = copy.deepcopy(old_replace_table)
    updated_rearrangement_table = copy.deepcopy(old_rearrange_table)

    while epochs != 0:
        old_rearrange_table = copy.deepcopy(updated_rearrangement_table)
        old_replace_table = copy.deepcopy(updated_replacement_table)
        flag = random.randint(
            0, 1
        )  # generate random number among 0 and 1 to modify one of the two encryption tables
        if flag == 0:
            updated_replacement_table = modify_encryption_table(old_replace_table, letters=True)
        else:
            updated_rearrangement_table = modify_encryption_table(old_rearrange_table)

        T_hat = encode_file(
            encoded_text, updated_replacement_table, updated_rearrangement_table
        )  # encoding with modified encryption tables
        T_hat_prob = get_probability_log(
            T_hat.split(" "), transition_probabilities, initial_probabilities
        )  # returns prob for guess with given encryption

        if T_hat_prob > probability_of_original_guess:  # new guess is better than old
            probability_of_original_guess = T_hat_prob
        else:  # new guess is not better than old
            if np.random.binomial(1, np.exp(T_hat_prob - probability_of_original_guess)) == 0:
                if flag == 1:  # changing one of the two encryption tables
                    updated_rearrangement_table = copy.deepcopy(old_rearrange_table)
                else:
                    updated_replacement_table = copy.deepcopy(old_replace_table)
            else:
                probability_of_original_guess = T_hat_prob

        epochs -= 1

    #    print(math.exp(T_hat_prob),math.exp(T_prob))
    return encode_file(encoded_text, updated_replacement_table, updated_rearrangement_table)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = read_clean_file(sys.argv[1])
    corpus = read_clean_file(sys.argv[2])

    # Start animation for decoding (why not?)
    t = threading.Thread(target=animate, args=(sys.argv[1], sys.argv[2]))
    t.start()

    # Start decoding
    tick = time.time()
    decoded_text = break_code(encoded, corpus)

    # Stop the animation
    decoded = True

    # Write to output file
    with open(sys.argv[3], "w") as file:
        print(decoded_text, file=file)

    # Print time taken to decode
    print("\nDone! Time taken: {0} seconds".format(time.time() - tick))
