from model import MyModel
import pandas as pd
import pickle

def store (X, y):
    X['probability'] = y
    print ('Result not saved to database.')


if __name__ == '__main__':
    X = pd.read_json('data/test_script_examples.json').head(1)

    with open('data/model.pkl', 'rb') as f:
        model = pickle.load(f)

    y = model.predict_proba(X)[0,1]
    print('{:.2f}%'.format(y*100))

    store (X, y)
