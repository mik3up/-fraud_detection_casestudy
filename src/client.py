from model import MyModel
import pandas as pd
import pickle
import requests
import time
from pymongo import MongoClient
import boto3


class EventAPIClient:
    """Realtime Events API Client"""

    def __init__(self, first_sequence_number=0,
                 api_url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/',
                 api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC',
                 db = None):
        """Initialize the API client."""
        self.next_sequence_number = first_sequence_number
        self.api_url = api_url
        self.api_key = api_key

        # Create mongo instance
        client = MongoClient('localhost', 27017)
        db = client['fraud']
        self.predictions = db['predictions']

        # Create an S3 client
        self.s3 = boto3.client('s3')

        # Load model
        with open('data/model.pkl', 'rb') as f:
            self.model = pickle.load(f)

    def save_to_database(self, row):
        """Save a data row to the database."""
        # Set row to pandas
        X = pd.DataFrame([row])
        # Predict
        y = self.model.predict_proba(X)
        # Append prediction
        row['probability'] = y[0,1].round(4)
        self.predictions.update(row, row, upsert=True)
        print('You have {} entries in your Database'.format(self.predictions.find().count()))

    def get_data(self):
        """Fetch data from the API."""
        payload = {'api_key': self.api_key,
                   'sequence_number': self.next_sequence_number}
        response = requests.post(self.api_url, json=payload)
        data = response.json()
        self.next_sequence_number = data['_next_sequence_number']
        return data['data']

    def collect(self, interval=30):
        """Check for new data from the API periodically."""
        while True:
            print("Requesting data...")
            data = self.get_data()
            if data:
                print("Saving...")
                for row in data:
                    self.save_to_database(row)
                ## Create csv image of database
                df =  pd.DataFrame(list(self.predictions.find()))
                df.to_csv('data/temp.csv', index=False)

                # Uploads the given file using a managed uploader, which will split up large
                # files automatically and upload parts in parallel.
                self.s3.upload_file('data/temp.csv', 'dsi-fraud-casestudy', 'live.csv', ExtraArgs={'ACL': 'public-read'})
            else:
                print("No new data received.")
            print(f"Waiting {interval} seconds...")
            time.sleep(interval)
