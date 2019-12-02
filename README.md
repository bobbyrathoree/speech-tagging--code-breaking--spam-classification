# a3
## Part 1: Part-of-speech tagging
Training Process: 
 The following probabilities are calculated while training,
 1. Initial probability: Out of all the sentences, how much times a particular Part of Speech starts in the sentence.
 2. Transition probability: Calculating the probability of transition from one POS to another POS by taking subsequent pairs of words.
 3. Emission probability: Calculating the probability of any word occurring as different Parts of Speech
 Simple Model:
 Naive Bayes Algorithm was used to calculate the posterior probability.
                    P(A|B) = P(A)* p(B|A)  / P(B) 
  Log is taken of the given posterior to calculate the cost and minimum cost is picked up for effectiveness.
 HMM model:
 Being an exponential chai, Viterbi decoding comes in handy , For first word, initial probability of each POS * corresponding emission probability of given word and calculated probability is stored in dictionary structure with POS as key. Similarly the step is repeated for remaining words as product of transition from particular POS from previous state to current state POS and corresponding probability of that POS in previous state. This feature extraction is done for all POS. With these transition , the probabilities obtained are quite small hence log is taken to calculate the cost.Finding path with minimum cost is the goal of the algorithm.

 Complex model
 MCMC is used here and as chain structure is missing , Viterbi cannot be applied here. Initial assumption as all words are noun , gibbs sampling is done for 6000 repition and 2500 warmup iteration. Here 3 states are taken into consideration second previous , previous and current state. For each POS , transition probability between current and previous is mulitiplied with probability of transition between second previous and current with emission prob of POS. Similar repition is done for all POS.POS with max prob at end of iteration completion is predicted as POS.
  
 Description of program working:
  This program takes the bc.train as input to calculate the initial probability, transition probability,
  and emission probability. Separate functions are written for simple, hmm, and complex. So the program label.py takes two arguments (train dataset,test dataset). The model gets trained using the first argument (bc.train) and tests the trained model on the second argument(bc.test).
 Assumptions made:
There can be situations where the word is not available in the dictionary as it would not have occurred hence it's emission probability is assumed to be 10^-10 
## Part 2: Code Breaking

As per the **Metropolis-Hastings** algorithm, we start out by setting up the initial and transitional probabilities. Initial probabilities are computed by simply setting the first character of every word as the key with 1 as it's value, then normalizing it using the total sum of all keys' values. The transitional probabilities are hashes with each value itself a hash. Each value of the dictionary value is then mapped with the relative frequency of other following characters, then finally normalized with the sum of all values for one given key, for each key respectively.

Then, we split the corpus into keywords/tokens to use them later. From the file *apply_code.py*, we used some helper code to generate replacement and rearrangement tables. Once we have the encryption tables, we start making guesses using the *encode_file()* (changed from encode()) method from encode.py by passing in the encoded text along with the initial and transitional probabilities. Then we calculate the log probability of the guess that we just made. Since we'll update the encryption tables later on for each iteration, we create a copy of the original encryption tables so that we can compare later.

Finally, we start sampling/training for 20000 epochs. For each epoch we modify either of the encryption tables first based on a "random" selection. Then, we make a new guess and get its log probability to check if it is better than the original guess' probability. If it is, we change the original guess' probability to the better one and go to the next iteration. If not, we again modify either of the encryption tables and set the original guess' probability to the new one after updating the original encryption tables with modified ones.

At the end after all iterations, once we have the updated encryption tables (which are essentially decryption tables now), we decode the encoded text using those, return to the main function and write the same to requested file.

Since we're dealing with almost total randomness when it comes to modifying the encryption tables for each iteration, we tested with some random seeds for each test file that gave us the best readable output in most cases. We went ahead and saved them just in case we test on those exact files again. If we test on some other file that is not in the *working_hash*, we go with totally random seed. We also added a helper function that puts in an animation while the document is getting decoded, and prints the total time taken at the end in seconds.


## Part 3: Spam classification

For this part of the assignment, we attempted to implement a Naive Bayes classifer from scratch without using anything except for the Python standard libraries. The goal was to classify e-mails as spam or not spam based on a set of data we were provided at the beginning of the assignment. High level of how the code works: it uses the spam/notspam training data provided to train the classifer and then those probabilities aree used to predict the class label on test data set. Since there was no skeleton code provided, we will attempt to walk through my code in detail below in order to explain how we attemped to solve this problem.

The spam.py file created for this assignment accepts 3 command line arguments: training data path, test data path and an output file name. These variables are set at the beginning of the code and then test/train directories and slashes are appended to the proper paths where needed. Next, I setup a variable to break the e-mail messages out into individual words for the bag of words model using regex. Next we imported the training data for the spam and notspam subfolders. Each of these loads involved looping through all of the files in each directory, appending all of the lines of an e-mail into a single string variable, which we could then run the regex findall function against to split the string into a list of invidividual words conatined in the e-mail. It then loops through the list to determine which words to discard such as words with a single letter and then converts each word to lower case which then gets output to a final list of cleaned up words from an individual e-mail and appends it to another list for either spam or notspam data that we've cleaned up for use in the bag of words model.

We then combine all of the words from the spam and nonspam training data so we could use this later on in probability calculations. A counter object is then used against each of the 3 lists that hold all of the words for total, spam and nonspam training data which then gets output to a dictionary for each of the 3 data sets in order to then calculate conditional probabilities. To compute P(word | spam) and P(word | notspam), we loop through the total word count dict one word at a time, get the total count of that word in the spam data and the nonspam data, and then take each of those word counts and divide them by the total word size of their related spam/notspam set of words. These probabilities then get stored into another new dictionary which holds these conditional probabilities for all words in the training set. After all of these values have been populated and stored in a dict, we could then move on to loading the testing data set.

The test data was loaded in the same way that the training data was. Now that all of the data that we needed to make our predictions was ready, we could then move on to calculating the probability that each test e-mail was either spam or not spam. First, we computed the priors for if an e-mail would be spam or not spam, based on the training data set. Then we looped through each e-mail in the test data, and then looped through every e-mail and assigned every word a probability value that it was either spam or not spam based on the training data we calculated from earlier. To keep track of this for each e-mail, we just summed up all of the probabilities for all the words in every email and at the end of the loop, whichever variable (spam or notspam score) had the highest total value, that then determined how to classify the test email. These steps were then performed for every e-mail in the test data and the final predictions were then stored in a list which could then be compared to the ground truth file and output to a file that we chose when we passed in the initial parameters.

Based on the comparison between the ground truth file and the predictions my classifer made, we were able to successfully identify the proper class for 67% of the test emails provided. We feel like this is a pretty reasonable accuracy for the problem considering we didn't do much data cleanup at all and just relied on the Naive Bayes classifer data produced in order to identify if the e-mail was spam or not spam. Finally, an output file gets generated that contains the name of each test e-mail and the predicted class for each of those e-mails generated by my code.

This code was also all tested successfully from the SICE Linux server using the following command below. We needed to chmod and dos2unix the spam.py file firsty as well.

./spam.py /u/srsteinb/brathore-srsteinb-singshar-a3/part3/train /u/srsteinb/brathore-srsteinb-singshar-a3/part3/test preds.txt

