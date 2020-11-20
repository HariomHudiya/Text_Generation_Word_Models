import numpy as np
import os
import string
import tensorflow as tf
from tensorflow.keras.layers import  Dense,LSTM,Embedding
from tensorflow.keras.models import Sequential
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from random import randint

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def file_reader(filepath,debugging_info = True):
    """

    Args:
        filepath: File path

    Returns: Read File

    """
    print("Reading File.............")

    file = open(filepath)
    text = file.read()    # read all the lines in the file # Just a continous single string
    file.close()
    print("File Accessed Successfully.........")

    if debugging_info:
        print("First 200 characters :\n\n", text[:200])

    return text

# clean verison does not have license verbose

filepath = "./datasets/republic_clean.txt"
data = file_reader(filepath)

print("----------------------------------------------")

## Preprocessing
# Replace ‘–‘ with a white space so we can split words better.
# Split words based on white space.
# Remove all punctuation from words to reduce the vocabulary size (e.g. ‘What?’ becomes ‘What’).
# Remove all words that are not alphabetic to remove standalone punctuation tokens.
# Normalize all words to lowercase to reduce the vocabulary size.

print("Preprocessing Starts..............")


# turn a doc into clean tokens
def clean_file(data):
    """
    
    Args:
        data: file containing a single string of text 

    Returns: cleaned and procesed file as tokens

    """""
    # replace '--' with a space ' '
    doc = data.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()    # the order of words is preserved
    # remove punctuation from each token
    # https://www.programiz.com/python-programming/methods/string/maketrans (for refrence)
    table = str.maketrans('', '', string.punctuation) # dict
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens

# clean document
tokens = clean_file(data)
print("First 200 Tokens: ")
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))
print("----------------------------------------")
def sequence_builder(tokens,seq_length = 50):
    """

    Args:
        tokens: Word in the corpora (list of words...)
        seq_length: That'll be taken as input

    Returns: seq_length +1 long sequences

    """
    print("Building Sequence....")
    seq_length = seq_length
    splitting_length = seq_length + 1
    sequences = []
    for i in range(splitting_length, len(tokens)):
        # select sequence of tokens
        seq = tokens[i - splitting_length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))
    return sequences
print("----------------------------------------")
sequences =sequence_builder(tokens)
print("Debugging Info for sequnce Builder")
print(np.array(sequences).shape)
print(sequences[5])           # Sring Object     # Replace 5 with any String number to check #51
print("Number of words in a sequnce : ",len([word for word in sequences[5].split(" ")]))

# save tokens to file, 51 tokens per line
def save_doc(lines, filepath):
    """
    
    Args:
        lines:  51 tokens
        filename: 

    Returns:

    """""
    print("Saving the File............")
    data = '\n'.join(lines)
    file = open(filepath, 'w')
    file.write(data)
    file.close()
    print("file saved.............")

filepath = "./datasets/republic_sequences.txt"
save_doc(sequences,filepath)

# Loading republic_sequences.txt from memory..

doc = file_reader(filepath)
lines = doc.split('\n')
print("10th Sequence")
print("Length of the Sequence", len([w for w in lines[9].split()]))
print(lines[9])

## Encoding Sequence
print("----------------------------------------------")




# Encode Sequences
# The word embedding layer expects input sequences to be comprised of integers.
#
# We can map each word in our vocabulary to a unique integer and encode our input sequences.
# Later, when we make predictions, we can convert the prediction to numbers and
# look up their associated words in the same mapping.
#
# To do this encoding, we will use the Tokenizer class in the Keras API.
#
# First, the Tokenizer must be trained on the entire training dataset,
# which means it finds all of the unique words in the data and assigns each a unique integer.
#
# We can then use the fit Tokenizer to encode all of the training sequences,
# converting each sequence from a list of words to a list of integers.

def sequence_encoder(lines):
    """

    Args:
        lines: 51 token "words" string

    Returns: 51 token "integer" list

    """
    print("Tokenising Sequence.............")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    return sequences,tokenizer

sequences,tokenizer = sequence_encoder(lines)
print("First Sequence : ")
print(sequences[0])
print("Length of first Sequence : ", len(sequences[0]))
print("Sequence Shape: ", np.array(sequences).shape)
print()

# We can access the mapping of words to integers as a dictionary attribute called word_index on the Tokenizer object.
#
# We need to know the size of the vocabulary for defining the embedding layer later.
# We can determine the vocabulary by calculating the size of the mapping dictionary.
#
# Words are assigned values from 1 to the total number of words (e.g. 7,409).
# The Embedding layer needs to allocate a vector representation for each word in this vocabulary
# from index 1 to the largest index and because indexing of arrays is zero-offset,
# the index of the word at the end of the vocabulary will be 7,409; that means the array must be 7,409 + 1 in length.
# So that last word is included , coz embedding will count from 0
#
# Therefore, when specifying the vocabulary size to the Embedding layer,
# we specify it as 1 larger than the actual vocabulary.

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("Vocab Size: ")
print(vocab_size)

def input_output_splitter(sequences,vocab_size):
    """

    Args:
        sequences: tokenized sequence of 51 integer tokens as a list

    Returns: X (50 integer token as list) , y (as one hot representation)

    """
    print("Splitting X and y ")
    sequences = np.array(sequences)
    X = sequences[:, :-1]
    y = sequences[:, -1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]  # 50
    print("Seq_length : ", seq_length)
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    return X,y

X,y = input_output_splitter(sequences,vocab_size)
seq_length = X.shape[1]

def model_builder(vocab_size,embedding_size = 50, seq_length = seq_length):
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    return  model

model = model_builder(vocab_size,embedding_size= 128,seq_length=seq_length)


# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model

checkpoint_dir = './plato_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

epochs = 10 # for training Purposes

#history = model.fit(dataset,epochs=epochs,callbacks=[checkpoint_callback])
print("Latest Checkpoiny ...")
print(tf.train.latest_checkpoint(checkpoint_dir))

# Save the Model and tokenizer

# save the model to file
model.save('model.h5')
# save the tokenizer
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

# # saving
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)



# Later, when we load the model to make predictions, we will also need the mapping of words to integers.
# This is in the Tokenizer object, and we can save that too using Pickle.

print("-------------------------------------------------------------------------------")
print("------------Generating Text Process Started----------------------")

def model_tokenizer_loader(model_path,tokenizer_path):
    """

    Args:
        model_path: Path of the Model to be loaded
        tokenizer_path: path of the TOkenizer to be used alongside the model

    Returns: Model , Tokenizer

    """
    print("Loading Model and Tokenizer")
    # load the model
    model = load_model(model_path)
    # load the tokenizer
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    print("Loading Successful.....")

    return model,tokenizer

tokenizer_path = "tokenizer.pkl"
model_path = "model.h5"


model,tokenizer = model_tokenizer_loader(model_path,tokenizer_path)

# select a seed text
def seed_text_selector(seed_filepath,file_reader):
    """

    Args:
        seed_filepath: file of the path from which seed text needs to be generated
        file_reader: Function that reads the file

    Returns: random seed_text(non-encoded)

    """
    lines = file_reader(seed_filepath, debugging_info=False)
    lines = lines.split('\n')
    seed_text = lines[randint(0, len(lines))]
    print("Seed Text: ")
    print(seed_text + '\n')
    print("Len of Seed Text : ", len([word for word in seed_text.split()]))

    return seed_text

def seed_text_encoder(seed_text,tokenizer):
    """

    Args:

        seed_filepath: file of the path from which seed text needs to be generated
        file_reader: Function that reads the file
        tokenizer: Tokenizer to encode seed text

    Returns: encoded seed_text

    """

    print("Encoding Seed Text: ")
    encoded = tokenizer.texts_to_sequences([seed_text])[0]
    print(encoded)

    return encoded

seed_filepath = './datasets/republic_sequences.txt'
seed_text = seed_text_selector(seed_filepath,file_reader)

encoded_seed_text = seed_text_encoder(seed_text,tokenizer)



# generate a sequence from a language model

def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    """

    Args:
        model: Model to be used to generate text
        tokenizer: Tokenizer to go along the model
        seq_length: Maximum length of Sequence to be generated
        seed_text: Any Random Text
        n_words: Number of Words to generate

    Returns:

    """
    print("Generating Text")
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        encoded_seed = tokenizer.texts_to_sequences([seed_text])[0]

        # truncate sequences to a fixed length
        encoded_seed = pad_sequences([encoded_seed], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        # The model can predict the next word directly by calling model.predict_classes()
        # that will return the index of the word with the highest probability.

        # yhat = model.predict_classes(encoded_seed, verbose=0)
        yhat = np.argmax(model.predict(encoded_seed), axis=-1)
        # map predicted word index to word
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

#seed_text = "Harry it is working !!!!!!! sapdm asdmaspdom sadmaposdmoasmd asd  "
#seed_text_len = len([word for word in seed_text.split()])
#
# WARNING:tensorflow:Model was constructed with shape (None, 50)
# for input Tensor("embedding_input_1:0", shape=(None, 50), dtype=float32),
# but it was called on an input with incompatible shape (None, 9).

generated_seq = generate_seq(model,tokenizer,50 , seed_text , 100)
print(generated_seq)
# print(len(generated_seq))
print(len([word for word in generated_seq.split()]))

########################### The probability is not getting updated as we progress.. that's why we are getting the same output #########

######## Update this after working on Language Models ###############################