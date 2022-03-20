import tensorflow as tf
import pandas as pd
from IPython.display import clear_output

'''
- We will predict the specie of Iris with sepal and petal measure.
- The model use DNNClassification with 2 hidden layers,
    30, 10 nodes respectively.
- There are 3 possible species.
- You can see the accuracy in the last row output.
'''


def data(CSV_PATH):
    '''
    Treating our dataframe
    '''
    header = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    df = pd.read_csv(CSV_PATH, names=header)

    # replacing species names (string) into (integer)
    for i, specie in enumerate(species):
        df['species'] = df['species'].replace(specie, i)
    
    # suffle the datafram then slice into two new datafram,
    # one to train (80%), the other to test (20%).
    df = df.sample(frac=1).reset_index(drop=True)
    df_train = df.sample(frac=0.8)
    df_eval = df.drop(df_train.index)
    y_train = df_train.pop('species')
    y_eval = df_eval.pop('species')

    return df_train, y_train, df_eval, y_eval


def input_fn(df_train, y_train, training=True, batch_size=256):
    ds = tf.data.Dataset.from_tensor_slices((dict(df_train), y_train))
    if training:
        ds = ds.shuffle(1000).repeat()

    return ds.batch(batch_size)


def mk_feature_columns(NUMERIC_COLUMNS):
    feature_columns = []
    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    return feature_columns


def main():
    CSV_PATH = 'https://query.data.world/s/hefzuesvkxqj3gibsxke4nahywcs22'
    NUMERIC_COLUMNS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        # the train and eval dataframe
    df_train, y_train, df_eval, y_eval = data(CSV_PATH)
    feature_columns = mk_feature_columns(NUMERIC_COLUMNS)
        # creating the model with 2 hidden layes, 30 and 10 nodes respectively.
    model = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[30, 10], n_classes=3)
        # training the model with 5000 steps
    model.train(input_fn=lambda: input_fn(df_train, y_train), steps=5000)
        # testing the model
    result = model.evaluate(lambda: input_fn(df_eval, y_eval, False))
        # printing the accuracy
    print('ACCURACY:', result['accuracy'])


if __name__ == "__main__":
    main()
