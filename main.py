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
max_length = 50
trunc_type = 'post'
padding_type = 'post'
embedding_dim = 16

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()



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

    #Final pre-processing
    # Before training, we still need to pad the sequences, as well as split into training and test sets.

    # Pad all sequences
    sequences_padded = pad_sequences(sentences, maxlen=max_length,
                                     padding=padding_type, truncating=trunc_type)

    # Separate out the sentences and labels into training and test sets
    training_size = int(len(sentences) * 0.8)

    training_sequences = sequences_padded[0:training_size]
    testing_sequences = sequences_padded[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    # Make labels into numpy arrays for use with the network later
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    # Create and train the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()

    num_epochs = 30
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(training_sequences, training_labels_final, epochs=num_epochs,
                        validation_data=(testing_sequences, testing_labels_final))

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")
















