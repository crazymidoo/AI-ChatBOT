##FILE DI ADDESTRAMENTO
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open('json/data.json', 'r') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        if not pattern.strip():
            continue
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in document[0]]
    for word in words:
        bag.append(1 if word in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

for bag, out in training:
    if len(bag) != len(words):
        print("ERRORE BAG SIZE:", len(bag), "!=", len(words))
    if len(out) != len(classes):
        print("ERRORE OUTPUT SIZE:", len(out), "!=", len(classes))

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('chatbot_model.keras')

print("Done!")
