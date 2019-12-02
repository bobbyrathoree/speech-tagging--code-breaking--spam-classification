#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: Scott Steinbruegge (srsteinb), Bobby Rathore (brathore), Sharad Singh (singshar)

import os
import re
import sys
from collections import Counter

# Accept train data folder, test data folder and output file name as command line parameters
train_data = sys.argv[1]
test_data = sys.argv[2]
output_file = sys.argv[3]

# Not sure if / will be including during input, so making sure it exists
if train_data[:-1] != "/":
    train_data += "/"

if test_data[:-1] != "/":
    test_data += "/"

# train_data = 'C:/Users/Scott/Documents/train'
# test_data = 'C:/Users/Scott/Documents/test'
# output_file = 'preds.txt'

# Break line into words w/ Regex
breaks = re.compile(r"[\w]+")

# Set load directories manually (change to input variable at end)
training_spam = train_data + "spam/"
training_notspam = train_data + "notspam/"


# Load training spam data
def load_data(path: str) -> list:
    result = list()
    for file in sorted(os.listdir(path)):
        if file != "cmds":
            opened_file = open(path + file, "r", encoding="Latin-1")
            file_data = ""

            for line in opened_file:
                file_data += line

            # Break each e-mail out into individual words for bag of words model
            words_in_file = breaks.findall(file_data)

            # Filter out data that's not helpful
            word_output = [
                word.lower()
                for word in words_in_file
                if len(word) > 1 and not word.isnumeric() and "_" not in word
            ]

            result.append(word_output)
    return result


spam = load_data(training_spam)
notspam = load_data(training_notspam)
test = load_data(test_data)

# Combine spam and notspam training data
train = spam + notspam

# Get word counts from the total, spam and notspam bag of words. Use these to calculate probabilities.
total_words = []
spam_words = []
notspam_words = []

for email in train:
    for word in email:
        total_words.append(word)

for email in spam:
    for word in email:
        spam_words.append(word)

for email in notspam:
    for word in email:
        notspam_words.append(word)

total_word_count = Counter(total_words)
spam_word_count = Counter(spam_words)
notspam_word_count = Counter(notspam_words)

# Train the model (calculate probabilities of words occuring in spam/notspam e-mails from the training data)
all_words_in_bow = len(total_words)
word_probs = {"spam": {}, "notspam": {}}

# P(word | spam) and P(word | notspam) to conditional probability table word_probs
for key, value in total_word_count.items():
    word_count_in_spam = spam_word_count[key]
    word_count_in_not_spam = notspam_word_count[key]
    word_probs["spam"][key] = word_count_in_spam / len(spam_words)  # spam_words
    word_probs["notspam"][key] = word_count_in_not_spam / len(
        notspam_words
    )  # notspam_words

# Now we can compute the prior and conditional probabilities using the training
# data and apply it to the test set to see if we can classify spam e-mails correctly
p_spam = len(spam) / len(train)
p_notspam = len(notspam) / len(train)

test_predictions = []

for email in test:
    total_spam_p = p_spam
    total_notspam_p = p_notspam

    for word in email:
        if word in word_probs["spam"]:
            total_spam_p += word_probs["spam"][word]

        if word in word_probs["notspam"]:
            total_notspam_p += word_probs["notspam"][word]

    if total_spam_p > total_notspam_p:
        test_predictions.append("spam")
    else:
        test_predictions.append("notspam")

# Check the accuracy of our predictions - Highest achieved = ~67%
opened_file = open("test-groundtruth.txt", "r", encoding="Latin-1")
truth_data = []

for line in opened_file:
    if " spam" in line:
        truth_data.append("spam")
    elif " notspam" in line:
        truth_data.append("notspam")

correct_preds = 0

for test, truth in zip(test_predictions, truth_data):
    if test == truth:
        correct_preds += 1

print("Prediction accuracy is:", correct_preds / len(truth_data))

# Output predictions for each test e-mail to a flat file
output = open(output_file, "w")

for file, preds in zip(sorted(os.listdir(test_data)), test_predictions):
    output.write(file + " " + preds + "\n")

output.close()

print("Output of test file predictions saved to :", output_file)
