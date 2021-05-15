from flask import Flask
import random
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os


# If `entrypoint` is not defined in app.yaml, App Engine will look for an app
# called `app` in `main.py`.
app = Flask(__name__)


@app.route('/ML')
def runML():
    ## Reading the given dataset
    spam = pd.read_csv('spam1.csv')

    ##print(spam.head())

    ## Converting the read dataset in to a list of tuples, each tuple(row) contianing the message and it's label
    data_set = []
    for index,row in spam.iterrows():
        data_set.append((row['message'], row['label']))

    ##### Preprocessing
    ##
    #### initialise the inbuilt Stemmer and the Lemmatizer
    stemmer = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()

    def preprocess(document, stem=True):
        'changes document to lower case, removes stopwords and lemmatizes/stems the remainder of the sentence'
    
        # change sentence to lower case
        document = document.lower()

        # tokenize into words
        words = word_tokenize(document)

        # remove stop words
        words = [word for word in words if word not in stopwords.words("english")]

        if stem:
            words = [stemmer.stem(word) for word in words]
        else:
            words = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]

        # join words to make sentence
        document = " ".join(words)

        return document

    #### - Performing the preprocessing steps on all messages
    messages_set = []
    for (message, label) in data_set:
        words_filtered = [e.lower() for e in preprocess(message, stem=False).split() if len(e) >= 3]
        messages_set.append((words_filtered, label))

    ##### Preparing to create features
    ##
    #### - creating a single list of all words in the entire dataset for feature list creation
    ##
    def get_words_in_messages(messages):
        all_words = []
        for (message, label) in messages:
          all_words.extend(message)
        return all_words

    ##
    #### - creating a final feature list using an intuitive FreqDist, to eliminate all the duplicate words
    #### Note : we can use the Frequency Distribution of the entire dataset to calculate Tf-Idf scores like we did earlier.
    ##
    def get_word_features(all_words):

        #print(wordlist[:10])
        wordlist = nltk.FreqDist(all_words)
        word_features = wordlist.keys()
        return word_features

    #### - creating the word features for the entire dataset
    word_features = get_word_features(get_words_in_messages(messages_set))

    ##### Preparing to create a train and test set
    ##
    #### - creating slicing index at 80% threshold
    # if splitPer == "80" :
    #     sliceIndex = int((len(messages_set)*.8))
    # elif splitPer == "70" :
    #     sliceIndex = int((len(messages_set)*.7))
    # elif splitPer == "60" :
    #     sliceIndex = int((len(messages_set)*.6))
    # else :
    #     sliceIndex = int((len(messages_set)*.5))

    #### - shuffle the pack to create a random and unbiased split of the dataset
    random.shuffle(messages_set)
    sliceIndex = int((len(messages_set)*.8))
    print("Slice Index : ", sliceIndex)

    train_messages, test_messages = messages_set[:sliceIndex], messages_set[sliceIndex:]
    ##

    ##### Preparing to create feature maps for train and test data
    #### creating a LazyMap of feature presence for each of the 8K+ features with respect to each of the SMS messages
    def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    #### - creating the feature map of train and test data
    ##
    training_set = nltk.classify.apply_features(extract_features, train_messages)
    testing_set = nltk.classify.apply_features(extract_features, test_messages)

    print('Training set size : ', len(training_set))
    print('Test set size : ', len(testing_set))

    ##
    ##### Training
    ##
    #### Training the classifier with NaiveBayes algorithm
    spamClassifier = nltk.NaiveBayesClassifier.train(training_set)

    ##### Evaluation
    ##
    #### - Analyzing the accuracy of the test set
    print("Traning set accuracy : ", nltk.classify.accuracy(spamClassifier, training_set))

    #### Analyzing the accuracy of the test set
    print("Testing set accuracy : ", nltk.classify.accuracy(spamClassifier, testing_set))
    ##
    #### Testing a example message with our newly trained classifier
    msg="You are the winner and won $300"
    return spamClassifier.classify(extract_features(msg.split()))
    ##    print('Classification result : ', spamClassifier.classify(extract_features(m.split())))
    ##    print("Input message : ", m)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END gae_python3_app]
# [END gae_python38_app]
