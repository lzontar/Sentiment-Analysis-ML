import keras
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt

from util.models.IClassifier import IClassifier


class LSTM_(IClassifier):
    def __init__(self, word_to_vec_map, n_splits=10, max_words=4000, epochs=100, plot=False, lr=0.00005):
        self.max_seq_len = 60
        self.label = 'LSTM'
        self.max_words = max_words
        self.epochs = epochs
        self.word_to_vec_map = word_to_vec_map
        self.model = None
        self.n_splits = n_splits
        self.random_state = 1
        self.tokenizer = Tokenizer(num_words=self.max_words, split=' ', oov_token='OOV')
        self.optimizer = Adam(learning_rate=lr)
        self.plot = plot
        self.batch_size = 64
        self.earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        self.modelCheckpoint = ModelCheckpoint(f'../data/results/models/LSTM{"Primary" if self.plot else ""}.h5',
                                               monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    def prepDataClassifier(self, X_train, y_train, X_test, y_test):
        whole_data = pd.concat([X_train, X_test])
        self.tokenizer.fit_on_texts(X_train.values)

        X_train_, y_train_ = self.prepDataClassifierSingle(X_train, y_train)
        X_test_, y_test_ = self.prepDataClassifierSingle(X_test, y_test)

        return X_train_, y_train_, X_test_, y_test_

    def prepDataClassifierSingle(self, X, y):
        X_ = self.tokenizer.texts_to_sequences(X.values)
        X_ = pd.DataFrame(pad_sequences(X_, maxlen=self.max_seq_len, padding='post'))
        y_ = y
        return X_, y_

    def fitClassifierInner(self, X_train, y_train, X_test, y_test):
        X_train_, y_train_, X_test_, y_test_ = self.prepDataClassifier(X_train, y_train, X_test, y_test)


        words_to_index = self.tokenizer.word_index
        vocab_len = len(words_to_index) + 1
        embed_vector_len = self.word_to_vec_map['moon'].shape[0]

        emb_matrix = np.zeros((vocab_len, embed_vector_len))

        for word, index in words_to_index.items():
            embedding_vector = self.word_to_vec_map.get(word)
            if embedding_vector is not None:
                emb_matrix[index, :] = embedding_vector

        embedding_layer = Embedding(
            vocab_len,
            embed_vector_len,
            embeddings_initializer=keras.initializers.Constant(emb_matrix),
            trainable=False,
        )

        self.model = Sequential()
        self.model.add(embedding_layer)
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(64)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer=self.optimizer,
                           metrics=['accuracy'])

        history = self.model.fit(X_train_,
                                 y_train_.values,
                                 validation_data=(X_test_, y_test_),
                                 epochs=self.epochs, batch_size=self.batch_size, callbacks=[self.earlyStopping, self.modelCheckpoint])

        epochs = len(history.history['loss'])
        if self.plot:
            df_history = pd.DataFrame({'Train loss': history.history['loss'], 'Test loss': history.history['val_loss'],
                                       'Train accuracy': history.history['accuracy'], 'Test accuracy': history.history['val_accuracy'],
                                       'Epoch': list(range(1, epochs + 1))})

            sns.lineplot(data=df_history, x='Epoch', y='Train loss', legend='brief', label='Train loss')
            sns.lineplot(data=df_history, x='Epoch', y='Test loss', legend='brief', label='Test loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("LSTM loss while learning")
            plt.legend()
            plt.savefig(f'../data/results/fig/lstm_loss_{epochs}.png')
            plt.show()
            plt.close()

            sns.lineplot(data=df_history, x='Epoch', y='Train accuracy', legend='brief', label='Train accuracy')
            sns.lineplot(data=df_history, x='Epoch', y='Test accuracy', legend='brief', label='Test accuracy')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("LSTM accuracy while learning")
            plt.legend()
            plt.savefig(f'../data/results/fig/lstm_accuracy_{epochs}.png')
            plt.show()
            plt.close()

    def predict(self, X, y):
        X_, y_ = self.prepDataClassifierSingle(X, y)
        return list(map(lambda x: int(x.item()), torch.round(torch.Tensor(self.model.predict(X_).squeeze()))))
