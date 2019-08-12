import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

class MyModel():
    def __init__(self):
        self.model = GradientBoostingClassifier(learning_rate=0.23,n_estimators=700)
        self.features = ["body_length","channels","delivery_method","fb_published",
          "org_facebook","org_twitter","user_age","has_header","venue_longitude",
          "payout_type2","user_type","user_created","name_length"]

    def fit(self, X, y):
        # Convert emails to labels
        # self.label = LabelEncoder()
        # email = self.label.fit_transform(X["email_domain"])
        # X["email_domain2"]= email
        # Map payout type to integers
        X["payout_type2"] = X["payout_type"].map({"ACH":1,"CHECK":2})
        X["payout_type2"].fillna(3,inplace = True)
        # Get only the predictive features
        X = X[self.features]
        # Fill missing value headers to False
        X["has_header"].fillna(0,inplace=True)
        # Fill remaining few missing with median
        self.median = X.median()
        X.fillna(self.median,inplace= True)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        # Convert emails to labels
        # email = self.label.transform(X["email_domain"])
        # X["email_domain2"]= email
        # Map payout type to integers
        X["payout_type2"] = X["payout_type"].map({"ACH":1,"CHECK":2})
        X["payout_type2"].fillna(3,inplace = True)
        # Get only the predictive features
        X = X[self.features]
        # Fill missing value headers to False
        X["has_header"].fillna(0,inplace=True)
        # Fill remaining few missing with median
        X.fillna(self.median,inplace= True)
        return self.model.predict_proba(X)


def get_data(datafile):
    df = pd.read_json(datafile)
    df['fraud'] = df['acct_type'].apply(lambda x: 'fraud' in x)
    y = df.pop('fraud')
    X = df
    return X, y

if __name__ == '__main__':
    X, y = get_data('data/data.json')
    model = MyModel()
    model.fit(X, y)
    with open('data/model.pkl', 'wb') as f:
        # Write the model to a file.
        pickle.dump(model, f)
