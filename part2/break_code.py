#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: Bobby Rathore
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
    :return: decoded text
    """
    # Get the initial and transitional probabilities
    initial_probabilities = get_initial_probabilities(corpus_text)
    transition_probabilities = get_transitional_probabilities(corpus_text)

    # Number of iterations to run the sampling for
    epochs = 20000
    number_of_iterations = 0

    # Get the initial encryption (replacement and rearrangement) tables
    original_replacement_table, original_rearrangement_table = get_initial_encryption_tables()

    # Make a guess
    original_guess = encode_file(encoded_text, original_replacement_table, original_rearrangement_table)
    probability_of_original_guess = get_probability_log(
        original_guess.split(" "), transition_probabilities, initial_probabilities
    )

    # Make a copy of each encryption table for later use
    updated_replacement_table = copy.deepcopy(original_replacement_table)
    updated_rearrangement_table = copy.deepcopy(original_rearrangement_table)

    while number_of_iterations <= epochs:
        original_rearrangement_table = copy.deepcopy(updated_rearrangement_table)
        original_replacement_table = copy.deepcopy(updated_replacement_table)

        # Based upon 2 "random" numbers, we decide which encryption table to modify
        random_choice = random.choice(
            [69, 420]
        )
        if random_choice == 69:
            updated_replacement_table = modify_encryption_table(original_replacement_table, letters=True)
        else:
            updated_rearrangement_table = modify_encryption_table(original_rearrangement_table)

        # Make a new guess and get its probability
        new_guess = encode_file(
            encoded_text, updated_replacement_table, updated_rearrangement_table
        )
        new_guess_probability = get_probability_log(
            new_guess.split(" "), transition_probabilities, initial_probabilities
        )

        # If new guess is better than original, make the original equal to new
        # Else change any of the two encryption tables
        if new_guess_probability > probability_of_original_guess:
            probability_of_original_guess = new_guess_probability
        else:
            if np.random.binomial(1, np.exp(new_guess_probability - probability_of_original_guess)) == 0:
                if random_choice == 420:
                    updated_rearrangement_table = copy.deepcopy(original_rearrangement_table)
                else:
                    updated_replacement_table = copy.deepcopy(original_replacement_table)
            else:
                probability_of_original_guess = new_guess_probability

        number_of_iterations += 1

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
