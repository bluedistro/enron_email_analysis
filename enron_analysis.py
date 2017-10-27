import os, sys, email
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set_style("whitegrid")
import wordcloud

# Network Analysis
import networkx as nx

# NLP
from nltk.tokenize.regexp import RegexpTokenizer

from subprocess import check_output

# print(check_output(['ls', 'input/']).decode("utf8"))

# Read data into dataframe

# (Analyzing 1000 emails first due to lack of memory)
email_df = pd.read_csv('input/emails.csv', nrows=1000)

# print(email_df.shape)

# view sample f the 1000
# print(email_df.head())

# view a single message
# print(email_df['message'][0])

def get_text_from_email(msg):
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())
    return ''.join(parts)


def split_email_addresses(line):
    # separate multiple email addresses
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x:x.strip(), addrs))
    else:
        addrs = None
    return addrs

# parse email into a list of email objects
messages = list(map(email.message_from_string, email_df['message']))
email_df.drop('message', axis=1, inplace=True)
# Get fields from parsed email fields
keys = messages[0].keys()

for key in keys:
    email_df[key] = [doc[key] for doc in messages]
# Parse content from emails
email_df['content'] = list(map(get_text_from_email, messages))
# split multiple email addresses
email_df['From'] = email_df['From'].map(split_email_addresses)
email_df['To'] = email_df['To'].map(split_email_addresses)

# Extract the root of file as user
email_df['user'] = email_df['file'].map(lambda x:x.split('/')[0])

del messages

# successful parsing message contents and fields
# print(email_df.head())

# print('shape of dataframe: ', email_df.shape)
for col in email_df.columns:
    pass
    # print(col, email_df[col].nunique())

# Set index and drop columns with two few values
email_df = email_df.set_index('Message-ID').drop(['file', 'Mime-Version', 'Content-Type',
                                                  'Content-Transfer-Encoding'], axis=1)

# Parse datetime
email_df['Date'] = pd.to_datetime(email_df['Date'], infer_datetime_format=True)
# print(email_df.dtypes)

# Find out when emails were sent as a plot (Years)
ax = email_df.groupby(email_df['Date'].dt.year)['content'].count().plot()
ax.set_xlabel('Year', fontsize=18)
ax.set_ylabel('N emails', fontsize=18)
# plt.show()

# Find out when emails were sent as a plot (Days of the week)
ax = email_df.groupby(email_df['Date'].dt.year)['content'].count().plot()
ax.set_xlabel('Day of week', fontsize=18)
ax.set_ylabel('N emails', fontsize=18)
# plt.show()

# Find out when emails were sent as a plot (Hours of the day)
ax = email_df.groupby(email_df['Date'].dt.year)['content'].count().plot()
ax.set_xlabel('Hour', fontsize=18)
ax.set_ylabel('N emails', fontsize=18)
# plt.show()

# find out who sent the most of mails

# count the word in the subject and content
tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')
email_df['subject_wc'] = email_df['Subject'].map(lambda x:len(tokenizer.tokenize(x)))
email_df['content_wc'] = email_df['content'].map(lambda x:len(tokenizer.tokenize(x)))

group_by_people = email_df.groupby('user').agg({
    'content': 'count',
    'subject_wc': 'mean',
    'content_wc':'mean'
})

group_by_people.rename(columns={'content': 'N emails',
                                'subject_wc': 'Subject word count',
                                'content_wc': 'Content word count'}, inplace=True)

# print(group_by_people.sort('N emails', ascending=False).head())


sns.pairplot(group_by_people.reset_index(), hue='user')
# plt.show()


# who sent the most emails to whom

# checking emails sent to single email addresses first, more important stuffs
sub_df = email_df[['From', 'To', 'Date']].dropna()
# print(sub_df.shape)

# drop emails sent to multiple email addresses
sub_df = sub_df.loc[sub_df['To'].map(len) == 1]
# print(sub_df.shape)


# actually view who sent what to who
sub_df = sub_df.groupby(['From', 'To']).count().reset_index()
# Unpack frozensets
sub_df['From'] = sub_df['From'].map(lambda x: next(iter(x)))
sub_df['To'] = sub_df['To'].map(lambda x: next(iter(x)))

# rename column and print  the first 10 of such email sendings
sub_df.rename(columns={'Date': 'count'}, inplace=True)
print(sub_df.sort_values(by='count', ascending=False).head(10))

# make a network of email senders and recipients
G = nx.from_pandas_dataframe(sub_df, 'From', 'To', edge_attr='count', create_using=nx.DiGraph())
print('Number of nodes: %d, Number of edges: %d' % (G.number_of_nodes(), G.number_of_edges()))

