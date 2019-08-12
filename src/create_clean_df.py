import pandas as pd

if __name__ == "__main__":
    df = pd.read_json('data/data.zip')
    df['fraud'] = df['acct_type'].apply(lambda x: 'fraud' in x)
    df = df[df['acct_type']!='spammer']
    df.to_pickle('data/labelled_dataframe.p')
