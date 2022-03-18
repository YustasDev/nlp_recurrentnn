import keras.metrics
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
import pdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import io
from tensorboard.plugins import projector
import tensorflow_datasets as tfds

vocab_size = 1000




if __name__ == '__main__':

    recognition_phrases = [
        'My favorite food is ice cream',
        'do you like ice cream too?',
        'My dog likes ice cream!',
        "your favorite flavor of icecream is chocolate",
        "chocolate isn't good for dogs",
        "your dog, your cat, and your parrot prefer broccoli",
        "ProgForce is leader in development of AI systems",
        "The burning question is: When will the fighting in Ukraine stop?",
        "They gave us free chocolate cake and didn't charge us any money",
        "He gave us weapons to kill and didn't charge us any money",
    ]



#========== we’ll use portions of (https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set)
# that contains both Amazon product and Yelp restaurant reviews =====================================>

    path = tf.keras.utils.get_file('reviews.csv',
                                    'https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P')
    print(path)

    # Read the csv file
    dataset = pd.read_csv(path)

    # Extract out sentences and labels
    sentences = dataset['text'].tolist()
    labels = dataset['sentiment'].tolist()

    # Print some example sentences and labels
    for x in range(6):
        print(sentences[x])
        print(labels[x])
        print("\n")

    # Create a subwords dataset
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(sentences, vocab_size, max_subword_length=5)

    # How big is the vocab size?
    print("Vocab size is ", tokenizer.vocab_size)

    # Check that the tokenizer works appropriately
    num = 5
    print(sentences[num])
    encoded = tokenizer.encode(sentences[num])
    print(encoded)

    # Separately print out each subword, decoded
    for i in encoded:
        print(tokenizer.decode([i]))

    # IMPORTANT!
    # we will create sequences that we will use for training, actually encoding each individual sentence
    # by breaking each word into subwords. This is equivalent to `text_to_sequences` with `Tokenizer`,
    # where each whole word is encoded in a sentence
    for i, sentence in enumerate(sentences):
        sentences[i] = tokenizer.encode(sentence)

    # Check the sentences are appropriately replaced
    print(sentences[5])









