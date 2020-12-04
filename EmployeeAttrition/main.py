import pandas as pd
import tensorflow as tf
from tensorflow import keras

DATA_URL = 'input/employee_attrition.csv'


def load_data():
    data = pd.read_csv(DATA_URL)
    print(data.head())
    return data


def correlation_analysis(df):
    print(df.corr()['Attrition'])


def preprocess(df):
    np_data = df.to_numpy().astype(float)
    x_train = np_data[:, 1:7]
    y_train = np_data[:, 7]
    y_train = tf.keras.utils.to_categorical(y_train, 2)
    return x_train, y_train


EPOCHS = 100
BATCH_SIZE = 100
N_HIDDEN = 128
VERBOSE = 1
VALIDATION_SPLIT = 0.2


def model(x_train, y_train):
    mdl = tf.keras.models.Sequential()
    mdl.add(keras.layers.Dense(N_HIDDEN, input_shape=(6,), name='DL1', activation='relu'))
    mdl.add(keras.layers.Dense(N_HIDDEN, name='DL2', activation='relu'))
    mdl.add(keras.layers.Dense(2, name='Final', activation='softmax'))

    mdl.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    mdl.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=VERBOSE,
            validation_split=VALIDATION_SPLIT)

    return mdl


def main():
    data = load_data()

    correlation_analysis(data)
    x_train, y_train = preprocess(data)
    print("X-Train Shape: ", x_train.shape)
    print("Y-Train Shape: ", y_train.shape)

    m = model(x_train, y_train)

    e = m.predict_classes([
        [40, 4, 20, 5, 4, 4],
        [111, 5, 85, 3, 2, 2],
        [31, 2, 15, 4, 1, 4],
        [61, 4, 24, 1, 4, 3],
        [77, 4, 35, 3, 1, 1],
        [81, 5, 7, 1, 2, 3],
        [113, 4, 112, 5, 4, 1],
        [101, 2, 48, 5, 1, 4],
        [45, 4, 22, 5, 3, 1],
        [25, 2, 2, 2, 3, 2],
        [97, 3, 15, 3, 2, 4]
    ])
    print("Will employees leave? ", e)


if __name__ == "__main__":
    main()
