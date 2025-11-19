import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

with open('json/data.json', 'r') as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

#Funzione per pulire le frasi
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#"Bag of Words", converte una frase in una lista di 0 e 1 che indica se c'è la parola o no
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag=[0] * len(words)
    for w in sentence_words:
        for i, word in  enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)