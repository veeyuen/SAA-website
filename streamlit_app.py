# streamlit_app.py

import streamlit as st
import csv
import pandas as pd
import numpy as np
import datetime

from matplotlib import pyplot as plt
import seaborn as sns

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from google.oauth2 import service_account
from google.cloud import storage

bucket_name = "singapore_athletics_association"
file_path = "consolidated.csv"

## Data preprocess and cleaning
def preprocess(i, string, metric):

    global OP

    l=['discus', 'throw', 'jump', 'vault', 'shot']

    string=string.lower()

    if any(s in string for s in l)==True:

        OP=float(str(metric))

    else:

        searchstring = ":"
        searchstring2 = "."
        substring=str(metric)
        count = substring.count(searchstring)
        count2 = substring.count(searchstring2)

        if count==0:
#            OP=float(substring)
            OP=substring



        elif (type(metric)==datetime.time or type(metric)==datetime.datetime):

            time=str(metric)
            h, m ,s = time.split(':')
            OP = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())


        elif (count==1 and count2==1):

            m,s = metric.split(':')
            OP = float(datetime.timedelta(minutes=int(m),seconds=float(s)).total_seconds())

        elif (count==1 and count2==2):

            metric = metric.replace(".", ":", 1)

            h,m,s = metric.split(':')
            OP = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())


        elif (count==2 and count2==0):

            h,m,s = metric.split(':')
            OP = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())


    return OP

# Clean each row of input file

def clean(data):

    for i in range(len(data)):

        rowIndex = data.index[i]

        input_string=data.iloc[rowIndex,2]
        metric=data.iloc[rowIndex,6]

        processed_output = preprocess(i, input_string, metric)

        data.loc[rowIndex, 'Metric'] = processed_output

    return data




# Create API client.
#credentials = service_account.Credentials.from_service_account_info(
#    st.secrets["gcp_service_account"]
#)
#client = storage.Client(credentials=credentials)

# Retrieve file contents.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
#@st.cache_data(ttl=600)
#def read_file(bucket_name, file_path):
#    bucket = client.bucket(bucket_name)
#    content = bucket.blob(file_path).download_as_string().decode("utf-8")
#   return content


#def hello_world(request):
#    # it is mandatory initialize the storage client
#    client = storage.Client()
#    #please change the file's URI
#    temp = pd.read_csv('gs://singapore_athletics_association/consolidated.csv', encoding='utf-8')
#    print (temp.head())
#    return f'check the results in the logs'



#table = read_file(bucket_name, file_path)
#table=hello_world(file_path)

#print("all ok")

#st.table(content)


#st.dataframe(dataframe.style.highlight_max(axis=0))


URL = ("https://storage.googleapis.com/singapore_athletics_association/consolidated.csv")

@st.cache(persist=True)

def load_data():

    client = storage.Client()
    data = pd.read_csv(URL, usecols = ['Date','Event', 'Name', 'Age', 'Team', 'Result', 'm/s', 'Competition',
              'Year D.O.B.', 'Info, if any', 'Metric'])
    return data

data = load_data()

#st.dataframe(data)

## Interactive dataframe filtering
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    #modify = st.checkbox("Add filters")

    #if not modify:
    #    return df

    #df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

#st.dataframe(filter_dataframe(data))


events = data['Event'].drop_duplicates()
event_choice = st.sidebar.selectbox('Select the event:', events)
dates = data["Date"].loc[data["Event"] == event_choice]
#date_choice = st.sidebar.selectbox('Date', dates)

#filter=data.loc[(data['Event']==event_choice) & (data['Date']==date_choice)]


start_date = st.sidebar.selectbox('Start Date', dates)
end_date = st.sidebar.selectbox('End Date', dates)

mask = ((data['Date'] > start_date) & (data['Date'] <= end_date) & (data['Event']==event_choice))

filter=data.loc[mask]

st.dataframe(filter)

metrics = filter['Metric']

fig, ax = plt.subplots()

plt.title("Distribution of Times/Distances")
ax = sns.histplot(data=filter, x='Metric', kde=True, color = "#b80606")

#ax = sns.distplot(filter)


st.pyplot(fig)

summary = metrics.describe()
st.write(summary)


# Upload csv

uploaded_file = st.file_uploader("Upload records via CSV file", accept_multiple_files=False)

if uploaded_file is not None:

    df_new=pd.read_csv(uploaded_file)
    st.dataframe(df_new)
    df_processed=clean(df_new)
    st.dataframe(df_processed)



## Upload csv into GCS
def upload(file):

    client = storage.Client()
    bucket = client.get_bucket('singapore_athletics_association')
    blob = bucket.get_blob('consolidated.csv')
    blob.download_to_filename('consolidated.csv')
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
