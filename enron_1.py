import os, sys, email, re
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
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from subprocess import check_output


class analyse_enron:

    # (Analyzing 1000 emails first due to lack of memory)
    def __init__(self, size=None):
        self.size = size
        self.email_df = pd.read_csv('input/emails.csv', nrows=self.size)
        messages = list(map(email.message_from_string, self.email_df['message']))
        self.email_df.drop('message', axis=1, inplace=True)
        # Get fields from parsed email fields
        keys = messages[0].keys()

        for key in keys:
            self.email_df[key] = [doc[key] for doc in messages]
        # Parse content from emails
            self.email_df['content'] = list(map(self.get_text_from_email, messages))
        # split multiple email addresses
        self.email_df['From'] = self.email_df['From'].map(self.split_email_addresses)
        self.email_df['To'] = self.email_df['To'].map(self.split_email_addresses)

        # Extract the root of file as user
        self.email_df['user'] = self.email_df['file'].map(lambda x:x.split('/')[0])

        del messages

    def view_file_detail(self):

        # print(check_output(['ls', 'input/']).decode("utf8"))
        # Read data into dataframe
        file_input = check_output(['ls', 'input/'].decode('utf8'))
        shape_of_emails = self.email_df.shape
        return file_input, shape_of_emails

    # get the names/ emails of all the workers
    def get_workers_detail(self):
        # self.parsing_mails()
        self.__init__()
        workers_names = set()
        workers_emails = set()
        for i in  range(self.size):
            workers_names.add(self.email_df['X-From'][i])
            workers_emails.add(next(iter(self.email_df['From'][i])))
        workers_names = list(workers_names)
        workers_emails = list(workers_emails)
        return workers_names, workers_emails

    # search for emails sent by a worker in enron by just typing part of his/her name
    def get_individual_email(self, name='phillip'):
        # self.parsing_mails()
        self.__init__()
        name = name.lower()
        subjects = []
        contents = []
        for i in range(self.size):
            full_sender_name = (self.email_df['X-From'][i]).lower()
            matcher = re.search(r'\b{}\b'.format(name), full_sender_name)
            # print(full_sender_name)
            if matcher:
                subjects.append(self.email_df['Subject'][i])
                contents.append(self.email_df['content'][i])
                # print(self.email_df['Subject'][i])

        return subjects, contents

    # search for keywords in emails sent by particular workers for more info.
    def search_individual_email(self, name='phillip', key_word='forecast'):
        target_contents = []
        key_word = key_word.lower()
        _, contents = self.get_individual_email(name=name)

        # if there are really contents in the lists, then go ahead and work
        if len(contents) != 0:

            for content in contents:
                full_content = str(content).lower()
                key_word_matcher = re.search(r'\b{}\b'.format(key_word), full_content)
                if key_word_matcher:
                    target_contents.append(content)

        return target_contents

    # def view_sample_mails(self, sample_id = 25):
    #     self.__init__()
    #     single_sample = self.email_df['message'][sample_id]
    #     msg = email.message_from_string(single_sample)
    #     variation = self.email_df['Subject'][sample_id]
    #     # get only the content of the email
    #     for part in msg.walk():
    #         if part.get_content_type() == 'text/plain':
    #             msg = part.get_payload()
    #     return single_sample, msg, variation

    def get_text_from_email(self, msg):
        parts = []
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                parts.append(part.get_payload())
        return ''.join(parts)


    def split_email_addresses(self, line):
        # separate multiple email addresses
        if line:
            addrs = line.split(',')
            addrs = frozenset(map(lambda x:x.strip(), addrs))
        else:
            addrs = None
        return addrs

    # successful parsing message contents and fields and demo show only the first five
    def view_parsed_mails(self,):
        # self.parsing_mails()
        self.__init__()
        parsed = self.email_df.head()
        return parsed

    def view_dataframe_shape(self,):
        # print('shape of dataframe: ', email_df.shape)
        # self.parsing_mails()
        self.__init__()
        for col in self.email_df.columns:
            output = col, self.email_df[col].nunique()
            # print(col, email_df[col].nunique())
            return output

    # Set index and drop columns with two few values
    def set_and_drop(self):
        # self.parsing_mails()
        self.__init__()
        self.email_df = self.email_df.set_index('Message-ID').drop(['file', 'Mime-Version', 'Content-Type',
                                                          'Content-Transfer-Encoding'], axis=1)
        return self.email_df

    def parse_time(self):
        self.email_df = self.set_and_drop()
        # Parse datetime
        self.email_df['Date'] = pd.to_datetime(self.email_df['Date'], infer_datetime_format=True)
        return self.email_df.dtypes
        # print(email_df.dtypes)

    def plot_and_view_timestamps(self):
        self.parse_time()
        # Find out when emails were sent as a plot (Years)
        ax = self.email_df.groupby(self.email_df['Date'].dt.year)['content'].count().plot()
        ax.set_xlabel('Year', fontsize=18)
        ax.set_ylabel('N emails', fontsize=18)
        plt.show()

        # Find out when emails were sent as a plot (Days of the week)
        ax = self.email_df.groupby(self.email_df['Date'].dt.dayofweek)['content'].count().plot()
        ax.set_xlabel('Day of week', fontsize=18)
        ax.set_ylabel('N emails', fontsize=18)
        plt.show()

        # Find out when emails were sent as a plot (Hours of the day)
        ax = self.email_df.groupby(self.email_df['Date'].dt.hour)['content'].count().plot()
        ax.set_xlabel('Hour', fontsize=18)
        ax.set_ylabel('N emails', fontsize=18)
        plt.show()

    # find out who sent the most of mails
    def subject_and_content_count(self):
        self.set_and_drop()
        # count the word in the subject and content
        tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')
        self.email_df['subject_wc'] = self.email_df['Subject'].map(lambda x:len(tokenizer.tokenize(x)))
        self.email_df['content_wc'] = self.email_df['content'].map(lambda x:len(tokenizer.tokenize(x)))

        group_by_people = self.email_df.groupby('user').agg({
            'content': 'count',
            'subject_wc': 'mean',
            'content_wc':'mean'
        })

        group_by_people.rename(columns={'content': 'N emails',
                                        'subject_wc': 'Subject word count',
                                        'content_wc': 'Content word count'}, inplace=True)

        # print(group_by_people.sort('N emails', ascending=False).head())
        return group_by_people.sort_values(by='N emails', ascending=False).head()

    def sns_plot(self):
        sns.pairplot(self.subject_and_content_count().reset_index(), hue='user')
        # sns.pairplot(group_by_people.reset_index(), hue='user')
        plt.show()

    # who sent the most emails to whom
    def email_sent_data(self):
        self.set_and_drop()
        # checking emails sent to single email addresses first, more important stuffs
        sub_df = self.email_df[['From', 'To', 'Date']].dropna()
        # print(sub_df.shape)

        # drop emails sent to multiple email addresses [because it might mostly contain
        # unwanted information]
        sub_df = sub_df.loc[sub_df['To'].map(len) == 1]
        # print(sub_df.shape)

        # actually view who sent what to who
        sub_df = sub_df.groupby(['From', 'To']).count().reset_index()
        # Unpack frozensets
        sub_df['From'] = sub_df['From'].map(lambda x: next(iter(x)))
        sub_df['To'] = sub_df['To'].map(lambda x: next(iter(x)))

        # rename column and print  the first 10 of such email sendings
        sub_df.rename(columns={'Date': 'count'}, inplace=True)
        # print(sub_df.sort_values(by='count', ascending=False).head(10))
        return sub_df.sort_values(by='count', ascending=False).head(10), sub_df

    # this method enables one to know the number of emails sent by the id entered and to whom
    def tracker(self, personnel_name = None):
        processed_above = list(self.email_sent_data())
        target_contents = []
        for item in processed_above:
            full_content = str(item).lower()
            key_word_matcher = re.search(r'\b{}\b'.format(personnel_name), full_content)
            if key_word_matcher:
                target_contents.append(item)
        return target_contents

    # make a network of email senders and recipients
    def network(self):
        _, sub_df = self.email_sent_data()
        G = nx.from_pandas_dataframe(sub_df, 'From', 'To', edge_attr='count', create_using=nx.DiGraph())
        # print('Number of nodes: %d, Number of edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
        return 'Number of nodes: %d, Number of edges: %d' % (G.number_of_nodes(), G.number_of_edges())

    def word_clouding(self):
        self.set_and_drop()
        # What the emails say in subject
        subjects = ' '.join(self.email_df['Subject'])
        fig, ax = plt.subplots(figsize=(16, 12))
        wc = wordcloud.WordCloud(width=800,
                                 height=600,
                                 max_words=200,
                                 stopwords=ENGLISH_STOP_WORDS).generate(subjects)
        ax.imshow(wc)
        ax.axis("off")

        # What the emails say in content
        contents = ' '.join(self.email_df.sample(1000)['content'])
        fig, ax = plt.subplots(figsize=(16, 12))
        wc = wordcloud.WordCloud(width=800,
                                 height=600,
                                 max_words=200,
                                 stopwords=ENGLISH_STOP_WORDS).generate(contents)
        ax.imshow(wc)
        ax.axis("off")

        plt.show()

# testing

ae = analyse_enron(size=5000)

target_contents = ae.search_individual_email(name='phillip', key_word='forecast')
groupee =ae.subject_and_content_count()
if len(target_contents) == 0:
    print('No keyword matched')
else:
    print(ae.tracker(personnel_name='phillip'))

# me, you, us = ae.view_sample_mails(sample_id=23)
# print(you)

