# Write a function called score_unigrams that takes three arguments:
#   - a path to a folder of training data 
#   - a path to a test file that has a sentence on each line
#   - a path to an output CSV file
#
# Your function should do the following:
#   - train a single unigram model on the combined contents of every .txt file
#     in the training folder
#   - for each sentence (line) in the test file, calculate the log unigram 
#     probability ysing the trained model (see the lab handout for details on log 
#     probabilities)
#   - write a single CSV file to the output path. The CSV file should have two
#     columns with headers, called "sentence" and "unigram_prob" respectively.
#     "sentence" should contain the original sentence and "unigram_prob" should
#     contain its unigram probabilities.
#
# Additional details:
#   - there is training data in the training_data folder consisting of the contents 
#     of three novels by Jane Austen: Emma, Sense and Sensibility, and Pride and Prejudice
#   - there is test data you can use in the test_data folder
#   - be sure that your code works properly for words that are not in the 
#     training data. One of the test sentences contains the words 'color' (American spelling)
#     and 'television', neither of which are in the Austen novels. You should record a log
#     probability of -inf (corresponding to probability 0) for this sentence.
#   - your code should be insensitive to case, both in the training and testing data
#   - both the training and testing files have already been tokenized. This means that
#     punctuation marks have been split off of words. All you need to do to use the
#     data is to split it on spaces, and you will have your list of unigram tokens.
#   - you should treat punctuation marks as though they are words.
#   - it's fine to reuse parts of your unigram implementation from HW3.

# You will need to use log and -inf here. 
# You can add any additional import statements you need here.
from math import log, inf


import csv
from pathlib import Path
def train_unigram_model(training_data):
    word_counts = {}
    for file_path in Path(training_data).glob('*.txt'):
        with open(file_path, 'r') as file:
            for line in file:
                words = line.strip().split()
                for word in words:
                    word = word.lower()
                    word_counts[word] = word_counts.get(word,0)+1
                    total_words = sum(word_counts.values())
                    unigram_model = {}
                    for word, count in word_counts.items():
                        unigram_model[word] = count/total_words
    return unigram_model
def score_sentence (unigram_model, sentence):
    words = sentence.strip().split()
    log_prob = 0
    for word in words:
        word = word.lower()
        probability = unigram_model.get(word,0)
        if probability == 0:
            return -inf
        log_prob += log(probability)
    return log_prob
def score_unigrams (training_data, sentence_test, output_csv):
    unigram_model = train_unigram_model(training_data)
    results = []
    with open(sentence_test, 'r') as file:
        for line in file:
            log_prob = score_sentence(unigram_model, line)
            results.append((line.strip(), log_prob))
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sentence", "unigram_prob"])
        writer.writerows(results)


# Do not modify the following line
if __name__ == "__main__":
    # You can write code to test your function here
    pass 
