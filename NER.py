#!/usr/bin/env python
# coding: utf-8

# <img align=center src="https://rhyme.com/assets/img/logo-dark.png"></img>
# <h2 align=center> Named Entity Recognition (NER) using LSTMs with Keras</h2>

# ### Task 1: Project Overview and Import Modules

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
plt.style.use("ggplot")

import tensorflow as tf
print('Tensorflow version:', tf.__version__)
print('GPU detected:', tf.config.list_physical_devices('GPU'))


# ### Task 2: Load and Explore the NER Dataset

# *Essential info about tagged entities*:
# - geo = Geographical Entity
# - org = Organization
# - per = Person
# - gpe = Geopolitical Entity
# - tim = Time indicator
# - art = Artifact
# - eve = Event
# - nat = Natural Phenomenon

# In[2]:


data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
data.head(20)


# In[4]:


print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())


# In[3]:


words = list(set(data["Word"].values))
words.append("ENDPAD")
num_words = len(words)


# In[5]:


tags = list(set(data["Tag"].values))
num_tags = len(tags)


# ### Task 3: Retrieve Sentences and Corresponsing Tags

# In[6]:


class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# In[7]:


getter = SentenceGetter(data)
sentences = getter.sentences


# In[8]:


sentences[0]


# ### Task 4: Define Mappings between Sentences and Tags

# In[9]:


word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


# In[10]:


word2idx


# ### Task 5: Padding Input Sentences and Creating Train/Test Splits

# In[11]:


plt.hist([len(s) for s in sentences], bins=50)
plt.show()


# In[12]:


from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words-1)

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# ### Task 6: Build and Compile a Bidirectional LSTM Model

# In[14]:


from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional


# In[15]:


input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words, output_dim=50, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)
model = Model(input_word, out)
model.summary()


# In[16]:


model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# ### Task 7: Train the Model

# In[17]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot.tf_keras import PlotLossesCallback


# In[18]:


get_ipython().run_cell_magic('time', '', '\nchkpt = ModelCheckpoint("model_weights.h5", monitor=\'val_loss\',verbose=1, save_best_only=True, save_weights_only=True, mode=\'min\')\n\nearly_stopping = EarlyStopping(monitor=\'val_accuracy\', min_delta=0, patience=1, verbose=0, mode=\'max\', baseline=None, restore_best_weights=False)\n\ncallbacks = [PlotLossesCallback(), chkpt, early_stopping]\n\nhistory = model.fit(\n    x=x_train,\n    y=y_train,\n    validation_data=(x_test,y_test),\n    batch_size=32, \n    epochs=3,\n    callbacks=callbacks,\n    verbose=1\n)')


# ### Task 8: Evaluate Named Entity Recognition Model

# In[19]:


model.evaluate(x_test, y_test)


# In[29]:


i = np.random.randint(0, x_test.shape[0]) #659
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))


# In[ ]:




