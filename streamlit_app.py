# streamlit_app.py

import streamlit as st
import csv
import pandas as pd
import numpy as np


from google.oauth2 import service_account
from google.cloud import storage

bucket_name = "singapore_athletics_association"
file_path = "consolidated.csv"


# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = storage.Client(credentials=credentials)

# Retrieve file contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)
def read_file(bucket_name, file_path):
    bucket = client.bucket(bucket_name)
    content = bucket.blob(file_path).download_as_string().decode("utf-8")
#    temp = pd.read_csv('gs://singapore_athletics_association/consolidated.csv', encoding='utf-8')

    return content


def hello_world(request):
    # it is mandatory initialize the storage client
    client = storage.Client()
    #please change the file's URI
    temp = pd.read_csv('gs://singapore_athletics_association/consolidated.csv', encoding='utf-8')
    print (temp.head())
    return f'check the results in the logs'



#table = read_file(bucket_name, file_path)
table=hello_world(file_path)

print("all ok")

st.table(content)


#st.dataframe(dataframe.style.highlight_max(axis=0))


## Upload csv into GCS
def upload(file):

    client = storage.Client()
    bucket = client.get_bucket('singapore_athletics_association')
    blob = bucket.get_blob('file')
    blob.download_to_filename('file')
    fields = ['Date', 'Event', 'Name', 'Age', 'Team', 'Result', 'm/s', 'Competition',
              'Year D.O.B.', 'Info, if any']

    with open(r'consolidated.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    blob = bucket.blob("consolidated.csv")
    blob.upload_from_filename("consolidated.csv")

# Upload dataframe into GCS as csv
#    df.to_csv()
#    bucket.blob('consolidated.csv').upload_from_string(df.to_csv(), 'text/csv')
