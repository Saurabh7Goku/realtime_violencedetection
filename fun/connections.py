import pymongo
import pandas as pd

def mongoDB(server):
    client = pymongo.MongoClient(server)
    db = client['violenceRecords']

    def csv_to_json(filename, header=None):
        data = pd.read_csv(filename)
        return data.to_dict(orient='Records')

    db.violence.insert_many(csv_to_json('file.csv'))