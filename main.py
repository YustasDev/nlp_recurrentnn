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



keras = tf.keras

# parameter determination
vocab_size = 2800
embedding_dim = 16
max_length = 50
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '0') for i in text])

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()




if __name__ == '__main__':

    sentences1 = [
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

    # Optionally set the max number of words to tokenize.
    # The out of vocabulary (OOV) token represents words that are not in the index.
    # Call fit_on_text() on the tokenizer to generate unique numbers for each word
    tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences1)

    # Examine the word index
    word_index = tokenizer.word_index
    print(word_index)

    # Get the number for a given word
    print('For "favorite" - index is  :' + str(word_index['favorite']))

    # Create sequences for the sentences
    #After you tokenize the words, the word index contains a unique number for each word.
    # However, the numbers in the word index are not ordered. Words in a sentence have an order.
    # So after tokenizing the words, the next step is to generate sequences for the sentences.
    sequences = tokenizer.texts_to_sequences(sentences1)
    print('sequences :' + str(sequences))

    # Try turning sentences that contain words that
    # aren't in the word index into sequences.
    # Add your own sentences to the test_data
    sentences2 = ["I like hot chocolate",
                  "Whether there is life on Mars, whether there is no life on Mars, it is not known to science",
                  "What do you think about it?"]

    sequences2 = tokenizer.texts_to_sequences(sentences2)
    print('sequences2 :' + str(sequences2))

    padded = pad_sequences(sequences)
    print("Padded Sequences: ")
    print(padded)

    # now specify a max length for the padded sequences
    padded = pad_sequences(sequences, maxlen=15)
    print("Padded Sequences with maxlen = 15: ")
    print(padded)

    # put the padding at the end of the sequences
    padded = pad_sequences(sequences, maxlen=15, padding="post")
    print("Padded Sequences with maxlen = 15 & padding='post': ")
    print(padded)

    # Limit the length of the sequences, you will see some sequences get truncated
    padded = pad_sequences(sequences, maxlen=5)
    print("Padded Sequences with maxlen = 5: ")
    print(padded)


    # Remind ourselves which number corresponds to the
    # out of vocabulary token in the word index
    print("<OOV> has the number", word_index['<OOV>'], "in the word index.")

    # Convert the test sentences to sequences
    test_seq = tokenizer.texts_to_sequences(sentences2)
    print("Test Sequence = ", test_seq)

    # Pad the new sequences
    padded = pad_sequences(test_seq, maxlen=10)
    print("\nPadded Test Sequence: ")

    # Notice that "1" appears in the sequence wherever there's a word
    # that's not in the word index
    print(padded)

#========== we’ll use portions of (https://www.kaggle.com/marklvl/sentiment-labelled-sentences-data-set)
# that contains both Amazon product and Yelp restaurant reviews =====================================>

    path = tf.keras.utils.get_file('reviews.csv',
                                    'https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P')
    print(path)

    # Read the csv file
    dataset = pd.read_csv(path)
    """
    # Review the first few entries in the dataset
    print(dataset.head(7))

    # Get the reviews from the text column
    reviews = dataset['text'].tolist()

    # Create the tokenizer, specify the OOV token, tokenize the text, then inspect the word index.
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(reviews)
    word_index = tokenizer.word_index
    print(len(word_index))
    print(word_index)

    sequences = tokenizer.texts_to_sequences(reviews)
    padded_sequences = pad_sequences(sequences, padding='post')

    # What is the shape of the vector containing the padded sequences?
    # The shape shows the number of sequences and the length of each one.
    print(padded_sequences.shape)

    # What is the first review?
    print(reviews[0])

    # Show the sequence for the first review
    print(padded_sequences[0])
    """

    sentences = dataset['text'].tolist()
    labels = dataset['sentiment'].tolist()

    # Separate out the sentences and labels into training and test sets
    training_size = int(len(sentences) * 0.8)

    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    # Make labels into numpy arrays for use with the network later
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    print(word_index['the'])
    print(reverse_word_index[2])

    print('Training word_index: ')
    print(word_index)

    print(padded[1])
    print(decode_review(padded[1]), len(decode_review(padded[1])))
    print(training_sentences[1])

#====================== Build a basic sentiment network ===============================================>
    # Note the embedding layer is first,
    # and the output is only 1 node as it is either 0 or 1 (negative or positive)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(.4),
        tf.keras.layers.Dense(10, activation='relu'),
        # tf.keras.layers.Dropout(.3),
        # tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    #model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    #model.summary()

    # early_stopping = keras.callbacks.EarlyStopping(patience=50)
    # lr_schedule = keras.callbacks.LearningRateScheduler(
    #     lambda epoch: (3e-5) / 2 ** (epoch / 200))
    # model_checkpoint = keras.callbacks.ModelCheckpoint(
    #     "NLP_model.h5", save_best_only=True)
    num_epochs = 50
    history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))
              #callbacks = [early_stopping, lr_schedule, model_checkpoint])
#======================================= Visualize the training graph =============================>
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

#=============================== ## Get files for visualizing the network ======================================>
# First get the weights of the embedding layer
    e = model.layers[0]
    weights = e.get_weights()[0]
    print(weights.shape) # shape: (vocab_size, embedding_dim)

    # Write out the embedding vectors and metadata
    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
    out_v.close()
    out_m.close()

    ## Download the files
    # try:
    #     from google.colab import files
    # except ImportError:
    #     pass
    # else:
    #     files.download('vecs.tsv')
    #     files.download('meta.tsv')

#==================== Predicting Sentiment in New Reviews =======================================>
    # Use the model to predict a review
    # Create the sequences
    padding_type='post'
    sample_sequences = tokenizer.texts_to_sequences(sentences1)
    fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)

    print('\nHOT OFF THE PRESS! HERE ARE SOME NEWLY MINTED, ABSOLUTELY GENUINE REVIEWS!\n')

    #model = keras.models.load_model("NLP_model.h5")
    forecast = model.predict(fakes_padded)

    # The closer the class is to 1, the more positive the review is deemed to be
    for x in range(len(sentences1)):
      print(sentences1[x])
      print(forecast[x])
      print('\n')

