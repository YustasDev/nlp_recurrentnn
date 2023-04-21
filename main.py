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

vocab_size = 1500
max_length = 100
trunc_type = 'post'
padding_type = 'post'
embedding_dim = 16
num_epochs = 30


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

# Define a function to take a series of reviews
# and predict whether each one is a positive or negative review
def predict_review(model, new_sentences, maxlen=max_length, show_padded_sequence=True):
    # Keep the original sentences so that we can keep using them later
    # Create an array to hold the encoded sequences
    new_sequences = []

    # Convert the new reviews to sequences
    for i, frvw in enumerate(new_sentences):
        new_sequences.append(tokenizer.encode(frvw))

    # Pad all sequences for the new reviews
    new_reviews_padded = pad_sequences(new_sequences, maxlen=max_length,
                                       padding=padding_type, truncating=trunc_type)

    forecasts = model.predict(new_reviews_padded)

    # The closer the class is to 1, the more positive the review is
    for x in range(len(new_sentences)):

        # We can see the padded sequence if desired
        # Print the sequence
        if (show_padded_sequence):
            print(new_reviews_padded[x])
        # Print the review as text
        print(new_sentences[x])
        # Print its predicted class
        print(forecasts[x])
        print("\n")

def fit_model_now (model, sentences) :
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  model.summary()
  history = model.fit(training_sequences, training_labels_final, epochs=num_epochs,
                      validation_data=(testing_sequences, testing_labels_final))
  return history

def plot_results (history):
  plot_graphs(history, "accuracy")
  plot_graphs(history, "loss")

def fit_model_and_show_results (model, sentences):
  history = fit_model_now(model, sentences)
  plot_results(history)
  predict_review(model, sentences)



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
    print(sequences_padded[5])
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
    model1 = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model1.summary()

    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model1.fit(training_sequences, training_labels_final, epochs=num_epochs,
                        validation_data=(testing_sequences, testing_labels_final))

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    predict_review(model1, recognition_phrases)
    model1.save('simpleDenseModel.h5', save_format='h5')

    """
#========================== Create a new model that uses a bidirectional LSTM ===================>

    # Define the model
    model_bidi_lstm = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile and train the model and then show the predictions for our extra sentences
    fit_model_and_show_results(model_bidi_lstm, recognition_phrases)
    
    

#============================ N- bidirectional LSTM LAYERS=============================>
    early_stopping = keras.callbacks.EarlyStopping(patience=10)
    lr_schedule = keras.callbacks.LearningRateScheduler(
        lambda epoch: (2e-4) / 2 ** (epoch / 200))
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        "LSTM2_model.h5", save_best_only=True)

    model_multiple_bidi_lstm = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,
                                                           return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        #tf.keras.layers.Dropout(.3),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dropout(.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_multiple_bidi_lstm.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
    history = model_multiple_bidi_lstm.fit(training_sequences, training_labels_final, epochs=num_epochs,
                         validation_data=(testing_sequences, testing_labels_final),
                         callbacks = [early_stopping, lr_schedule, model_checkpoint])

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    model = keras.models.load_model("LSTM2_model.h5")
    predict_review(model, recognition_phrases)



    #fit_model_and_show_results(model_multiple_bidi_lstm, recognition_phrases)
    """
#==================== try other sentences =================================>
    #
    # my_reviews =["lovely", "dreadful", "stay away",
    #              "everything was hot exactly as I wanted",
    #              "everything was not exactly as I wanted",
    #              "they gave us free chocolate cake",
    #              "I've never eaten anything so spicy in my life, my throat burned for hours",
    #              "for a phone that is as expensive as this one I expect it to be much easier to use than this thing is",
    #              "we left there very full for a low price so I'd say you just can't go wrong at this place",
    #              "that place does not have quality meals and it isn't a good place to go for dinner",
    #              ]
    #
    # print("===================================\n", "Embeddings only:\n", "===================================", )
    # predict_review(model1, my_reviews, show_padded_sequence=False)
    #
    # print("===================================\n", "With a single bidirectional LSTM:\n", "===================================")
    # predict_review(model_bidi_lstm, my_reviews, show_padded_sequence=False)
    #
    # print("===================================\n", "With two bidirectional LSTMs:\n",  "===================================")
    # predict_review(model_multiple_bidi_lstm, my_reviews, show_padded_sequence=False)




