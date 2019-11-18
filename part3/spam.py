import os
import re
from collections import Counter

train_data = 'C:/Users/Scott/Documents/train'
test_data = 'C:/Users/Scott/Documents/test'

# Break line into works w/ Regex
breaks = re.compile(r'[\w]+')

# Clean up HTML tags
html_cleanup =  re.compile(r'<[^>]+>')

# Set load directorys manually (change to input variable at end)
training_spam = train_data + '/spam/'
training_notspam = train_data + '/notspam/'

spam = []
notspam = []

# Load training spam data
for file in os.listdir(training_spam):
    if file != "cmds":
        opened_file = open(training_spam + file, 'r', encoding="Latin-1")
        file_data = ''
        word_output = []
        
        for line in opened_file:
            file_data += line
        
        # Cleanup the HTML tag data in the email            
        file_data = html_cleanup.sub('', file_data)
        
        # Break each e-mail out into inidividual words for bag of words model
        words_in_file = breaks.findall(file_data)

        # Filter out data that's not helpful
        for word in words_in_file:
            if len(word) > 1 and not word.isnumeric() and not "_" in word:
                # Make all words lower case
                word_output.append(word.lower())
        
        spam.append(word_output)

# Load training notspam data
for file in os.listdir(training_notspam):
    if file != "cmds":
        opened_file = open(training_notspam + file, 'r', encoding="Latin-1")
        file_data = ''
        word_output = []
        
        for line in opened_file:
            file_data += line
        
        # Cleanup the HTML tag data in the email            
        file_data = html_cleanup.sub('', file_data)
        
        # Break each e-mail out into inidividual words for bag of words model
        words_in_file = breaks.findall(file_data)

        # Filter out data that's not helpful
        for word in words_in_file:
            if len(word) > 1 and not word.isnumeric() and not "_" in word:
                # Make all words lower case
                word_output.append(word.lower())
        
        notspam.append(word_output)

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

# Train the model (calculate probabilities of words occuring in spam/notspam e-mails)
all_words_in_bow = len(total_words)
word_probs = {'spam': {}, 'notspam': {}}
 
# P(word | spam) and P(word | notspam) to conditional probability table word_probs
for key, value in total_word_count.items():
    word_count_in_spam = spam_word_count[key]
    word_count_in_not_spam = notspam_word_count[key]
    word_probs['spam'][key] = word_count_in_spam / len(spam_words)
    word_probs['notspam'][key] = word_count_in_not_spam / len(notspam_words)

# Now we can use the prior and conditional probabilities on the test set to see if we can classify spam e-mails correctly

    
    

