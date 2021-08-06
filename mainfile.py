import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('stopwords')
from nltk.corpus import stopwords
from utils.attention import AttentionLayer 
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from bs4 import BeautifulSoup
import numpy as np

def read_data():
    """Read the data."""
    news1_df = pd.read_csv('/content/drive/MyDrive/ColabNotebooks/newfolder/data/news_summary.csv', encoding='latin-1', usecols=['headlines', 'text'])
    news2_df = pd.read_csv('/content/drive/MyDrive/ColabNotebooks/newfolder/data/morenews.csv', encoding='latin-1')

    return pd.concat([news1_df, news2_df], axis=0).reset_index(drop=True)


df = read_data()

#df = df.iloc[:int(df.shape[0]*1),:]

df.head()


#df['Summary']=df.Summary.apply(lambda x: x.lower())
#df['Text']=df.Text.apply(lambda x: x.lower())

stop_words=stopwords.words('english')
print(stop_words)


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}


def text_cleaner(text):
       # convert everything to lower case
        cleaned = text.lower()
       # remove HTML tags
        cleaned = BeautifulSoup(cleaned,'html.parser').text
       # remove ('s), punctuations, special chars and any text inside parenthesis
        cleaned = re.sub(r"\([^)]*\)", "", cleaned)
        cleaned = re.sub('"', '', cleaned)
        cleaned = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in cleaned.split(" ")])    
        cleaned = re.sub(r"'s\b","",cleaned)
        cleaned = re.sub("[^a-zA-Z]", " ", cleaned) 
        tokens = [w for w in cleaned.split() if w not in stop_words]
        long_words = []
        for i in tokens:
              if len(i) >= 2:
                     long_words.append(i)
        return (" ".join(long_words)).strip()


cleaned_text = [text_cleaner(text) for text in df['text']]

def summary_cleaner(text):
            cleaned = re.sub("[^a-zA-Z]", " ", text)
            cleaned=cleaned.lower()
            cleaned = re.sub('"', '', cleaned)
            cleaned = " ".join([contraction_mapping[t] if t in contraction_mapping else t for t in cleaned.split(" ")])
            cleaned = re.sub(r"'s\b", "", cleaned)
            tokens = cleaned.split()
            cleaned = ""
            for i in tokens:
                  if len(i) >= 2:
                         cleaned = cleaned + i + " "
            return cleaned


cleaned_summary = [summary_cleaner(text) for text in df['headlines']]

df['cleaned_text'] = cleaned_text
df['cleaned_summary'] = cleaned_summary
#df['cleaned_summar'].replace('', np.nan, inplace = True)    # replace empty strings with NA
df.dropna(axis = 0, inplace = True)                          # remove rows that are NA 
df['cleaned_summary'] = df['cleaned_summary'].apply(lambda x: 'startsum ' + x + ' endsum')
df.reset_index(drop = True, inplace = True)


# EDA for the preprocessed data
length_of_texts = []
length_of_summaries = []

for i in range(len(df.index)):
       length_of_texts.append(len(df.at[i, 'cleaned_text'].split()))
       length_of_summaries.append(len(df.at[i, 'cleaned_summary'].split()))


print(length_of_texts[:3]," ",length_of_summaries[:3])


df['cleaned_text']

train_x, test_x, train_y, test_y = train_test_split(df['cleaned_text'], 
                                                 df['cleaned_summary'], 
                                                 test_size = 0.1, 
                                                 random_state = 42, 
                                                 shuffle = True)

text_tokenizer_text = Tokenizer()
text_tokenizer_headline=Tokenizer()
text_tokenizer_text.fit_on_texts(df['cleaned_text'])
text_tokenizer_headline.fit_on_texts(df['cleaned_summary'])


text_tokenizer_headline.word_index


x_voc = len(text_tokenizer_text.word_index) + 1
y_voc = len(text_tokenizer_headline.word_index) + 1

y_train_seq    =   text_tokenizer_headline.texts_to_sequences(train_y) 
y_test_seq   =   text_tokenizer_headline.texts_to_sequences(test_y)

max_len_text  = 60
max_len_headlines = 15

#padding zero upto maximum length
y_train    =   pad_sequences(y_train_seq, maxlen=max_len_headlines, padding='post')
y_test   =   pad_sequences(y_test_seq, maxlen=max_len_headlines, padding='post')

#convert text sequences into integer sequences (i.e one-hot encodeing all the words)
x_train_seq    =   text_tokenizer_text.texts_to_sequences(train_x) 
x_test_seq   =   text_tokenizer_text.texts_to_sequences(test_x)

#padding zero upto maximum length
x_tr    =   pad_sequences(x_train_seq,  maxlen=max_len_text, padding='post')
x_val   =   pad_sequences(x_test_seq, maxlen=max_len_text, padding='post')

num_words_x= len(text_tokenizer_text.word_index)+1
num_words_y = len(text_tokenizer_headline.word_counts)+1

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten,Embedding
from tensorflow.keras.layers import LSTM, Bidirectional,Concatenate
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential, Model
K.clear_session() 
latent_dim = 500 

# Encoder 
encoder_inputs = Input(shape=(max_len_text,)) 
enc_emb = Embedding(num_words_x, 300,trainable=True)(encoder_inputs) 

encoder_lstm_layer = Bidirectional(LSTM(latent_dim, return_state=True, return_sequences = True, 
                                             name = 'lstm_encoder'), merge_mode = 'concat')

encoder_output, forward_h, forward_c, backward_h, backward_c = encoder_lstm_layer(enc_emb)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

# Set up the decoder. 
decoder_inputs = Input(shape=(None,)) 
dec_emb_layer = Embedding(num_words_y,latent_dim, input_length=y_train.shape[1],trainable  = True) 
dec_emb = dec_emb_layer(decoder_inputs) 

#LSTM using encoder_states as initial state
decoder_lstm_layer = LSTM(2*latent_dim, return_state=True, return_sequences = True, name = 'lstm_decoder')
decoder_output , decoder_h, decoder_c = decoder_lstm_layer(dec_emb,
                                                   initial_state = encoder_states)

attn_layer = AttentionLayer(name = 'attention_layer')

######### -------------Attention layer---------------------------
attn_out, attn_states = attn_layer([encoder_output, decoder_output])
decoder_concat_input = Concatenate(axis=-1, name='concat')([decoder_output, attn_out])



#Dense layer
decoder_dense = TimeDistributed(Dense(num_words_y, activation='softmax')) 
decoder_outputs = decoder_dense(decoder_concat_input) 


model = Model([encoder_inputs,decoder_inputs], decoder_outputs) 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')                                                          
# using sparse_categorical entropy will solve memory problem                                                          
model.summary()


es = EarlyStopping(monitor='val_loss',patience = 10, mode='min', verbose=1)
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 1, mode = 'min', verbose = 1)


history=model.fit([x_tr,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=10,
                  callbacks=[es,lr],
                  batch_size=128, validation_data=([x_val,y_test[:,:-1]], y_test.reshape(y_test.shape[0],y_test.shape[1], 1)[:,1:]))

model.save('my_model.h5')
model.save_weights('model_weights.h5')
# Define inference model
encoder_model = Model(encoder_inputs, [encoder_output,state_h, state_c])


# now lets design our decoder model 
decoder_state_input_h = Input(shape=(2*latent_dim,))  # These states are required for feeding back to our next timestep decoder
decoder_state_input_c = Input(shape=(2*latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_hidden_state_input = Input(shape=(max_len_text,2*latent_dim)) # since we are using bidirectional lstm


# Get the embeddings of the decoder sequence
dec_emb= dec_emb_layer(decoder_inputs) 
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_output, state_h2, state_c2 = decoder_lstm_layer(dec_emb, initial_state=decoder_states_inputs)

attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_output])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_output, attn_out_inf])


# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
[decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
[decoder_outputs] + [state_h2, state_c2])

text_tokenizer_headline.index_word

# defined a new variable to change words2index nd index2words
reverse_target_word_index=text_tokenizer_headline.index_word
reverse_source_word_index=text_tokenizer_text.index_word
target_word_index=text_tokenizer_headline.word_index

# function for prediction of whole sentence by using loop
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out,e_h, e_c = encoder_model.predict(input_seq)
    
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['startsum']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out,e_h, e_c])
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token=reverse_target_word_index[sampled_token_index]
        
        if(sampled_token!='endsum'):
            decoded_sentence += ' '+sampled_token
        
        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'endsum'  or len(decoded_sentence.split()) >= (max_len_headlines-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['startsum']) and i!=target_word_index['endsum']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

for i in range(1,100):
    print("Review:",seq2text(x_val[i]))
    print("Original summary:",seq2summary(y_test[i]))
    print("Predicted summary:",decode_sequence(x_val[i].reshape(1,max_len_text)))
    print("\n")



