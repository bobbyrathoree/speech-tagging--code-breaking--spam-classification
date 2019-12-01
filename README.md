# a3
## Part 1
Training: 
 The following probabilities are calculated while training the dataset,
 1. Initial state probabilities: Out of all the sentences, how much times a particular P of Speech starts the sentence.
 2. Transition probabilities: Calculate the transition from one POS to another POS by taking subsequent
                              pairs of words.
 3. Emission probabilities: Calculate the probability of a any word occuring as different Parts of Speech

 Simplified model abstraction:
  We used Naive Bayes method for our simplified model. Naive Bayes equation for calculating posterior probability
  is as follows,
                    P(A|B) = p(B|A) * P(A) / P(B)
  Here, P(B)(Probability of a given word) is common among all words so we eliminate it in our calculations.
  The calculated P(B) will be proportional to the actual probabilities. For calculating the probability of a
  POS for a word, the emission probability of the POS given word is multiplied by the probability of the
  POS occuring in the training data. Natural logarithms are taken for the calculated probabilities which converts
  the probabilities to cost. For predicting the POS of a word we take the POS which has minimum cost.  

 HMM model abstraction:
  As the HMM model has a chain structure, Viterbi decoding is used for predicting the parts of speech of
  words. For calculating the POS probabilities of the first word the initial probability of each pos is
  multiplied with their corresponding emission probabilities given the word and the calculated probability
  is stored in a dictionary with the POS as the key. And to calculate the POS probabilities of the remaining
  words the probability of transition from a particular POS of previous state to a particular POS of current
  state is multiplied with the probability of the that POS in previous state. This is done for all POS in 
  previous state and the maximum is taken and multiplied with the emission probability of that POS given word.
  The same is repeated for all POS for the current state and the probabilities are stored in a dictionary.
  As the probabilities are too low natural logarithm is taken to convert them to cost, higher the probability
  lower the cost. The correct sequence of words is predicted by backtracking from the last word to the first 
  word using the  path with minimum cost.

 Complex model abstraction:
  As the complex model doesnot have a chain structure Markov Chain Monte Carlo is used instead of Viterbi 
  algorithm. Initially all the words are assumed to be noun and gibbs sampling is done for 5000 iterations 
  with warmup iterations as 4000. For calculating the probabilities of each pos for gibbs sampling the 
  transition probabilities between the current state and previous state is multiplied with the transition 
  probability between the current state and the state before the previous state and the emission probability
  of a POS. The same is calculated for all POS for the word and given as the probability for sampling.
  The POS with the max probability at the end of iteration is predicted as the POS.
  
 Description of program working:
  This program takes the bc.train as input to calculate the initial state probabilities, transition probabilities,
  and emission probabilities using the tuple combination of word and POS provided by the skeleton code. The 
  probabilities are stored in seperate dictionaries. Seperate functions are written for Simple, HMM, and complex 
  model. So the program label.py takes two arguments (training dataset and test dataset). The model gets trained 
  using the first argument (bc.train) and tests the trained model on the second argument(bc.test).

 Assumptions made:
  1) It is obvious that emission probabilities of a POS will not have all the words in English language, in 
     instances where a word is not in the dictionary of the POS for emission probability the probability is
     assumed to be 10^-10.
 Word and sentency accuracy achieved with the bc.test dataset:
                      Words correct:     Sentences correct: 
      Ground truth:      100.00%              100.00%
            Simple:       97.62%               66.67%
               HMM:       97.84%               66.86%
           Complex:       92.48%               45.35%
                            
                               

## Part 2
## Part 3 - Scott Steinbruegge

For this part of the assignment, I attemped to implement a Naive Bayes classifer from scratch without using anything except for the Python standard libraries. The goal was to classify e-mails as spam or not spam based on a set of data we were provided at the beginning of the assignment. High level of how the code works: it uses the spam/notspam training data provided to train the classifer and then those probabilities aree used to predict the class label on test data set. Since there was no skeleton code provided, I will attempt to walk through my code in detail below in order to explain how I attemped to solve this problem.

The spam.py file created for this assignment accepts 3 command line arguments: training data path, test data path and an output file name. These variables are set at the beginning of the code and then test/train directories and slashes are appended to the proper paths where needed. Next, I setup a variable to break the e-mail messages out into individual words for the bag of words model using regex. Next I imported the training data for the spam and notspam subfolders. Each of these loads involved looping through all of the files in each directory, appending all of the lines of an e-mail into a single string variable, which I could then run the regex findall function against to split the string into a list of invidividual words conatined in the e-mail. It then loops through the list to determine which words to discard such as words with a single letter and then converts each word to lower case which then gets output to a final list of cleaned up words from an individual e-mail and appends it to another list for either spam or notspam data that I've cleaned up for use in the bag of words model.

I then combine all of the words from the spam and nonspam training data so I can use this later on in probability calculations. A counter object is then used against each of the 3 lists that hold all of the words for total, spam and nonspam training data which then gets output to a dictionary for each of the 3 data sets in order to then calculate conditional probabilities. To compute P(word | spam) and P(word | notspam), I loop through the total word count dict one word at a time, get the total count of that word in the spam data and the nonspam data, and then take each of those word counts and divide them by the total word size of their related spam/notspam set of words. These probabilities then get stored into another new dictionary which holds these conditional probabilities for all words in the training set. After all of these values have been populated and stored in a dict, I could then move on to loading the testing data set.

The test data was loaded in the same way that the training data was. Now that all of the data that we needed to make my predictions was ready, I could then move on to calculating the probability that each test e-mail was either spam or not spam. First, I computed the priors for if an e-mail would be spam or not spam, based on the training data set. Then I looped through each e-mail in the test data, and then looped through every e-mail and assigned every word a probability value that it was either spam or not spam based on the training data we calculated from earlier. To keep track of this for each e-mail, I just summed up all of the probabilities for all the words in every email and at the end of the loop, whichever variable (spam or notspam score) had the highest total value, that then determined how to classify the test email. These steps were then performed for every e-mail in the test data and the final predictions were then stored in a list which could then be compared to the ground truth file and output to a file that we chose when we passed in the initial parameters.

Based on the comparison between the ground truth file and the predictions my classifer made, I was able to successfully identify the proper class for 67% of the test emails provided. I feel like this is a pretty reasonable accuracy for the problem considering I didn't do much data cleanup at all and just relied on the Naive Bayes classifer data produced in order to identify if the e-mail was spam or not spam. Finally, an output file gets generated that contains the name of each test e-mail and the predicted class for each of those e-mails generated by my code.

This code was also all tested successfully from the SICE Linux server using the following command below. I needed to chmod and dos2unix the spam.py file firsty as well.

./spam.py /u/srsteinb/brathore-srsteinb-singshar-a3/part3/train /u/srsteinb/brathore-srsteinb-singshar-a3/part3/test preds.txt

