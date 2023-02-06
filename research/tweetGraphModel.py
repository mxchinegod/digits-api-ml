import re
import os
import csv
import time
import json
import pytz
import torch
import tweepy
import warnings
import termcolor
import numpy as np
import configparser
import pandas as pd
import networkx as nx
from datetime import datetime
from collections import Counter
from playsound import playsound
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel


warnings.filterwarnings("ignore", category=UserWarning)


def rainbow_print(text):
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']
    n = len(text)
    for i in range(n):
        for j in range(20):
            char = text[(i + j) % n]
            color = colors[j % len(colors)]
            print(termcolor.colored(char, color), end='', flush=True)
        time.sleep(0.1)
        print('\033[2K\033[1G', end='', flush=True)  # erase line


_DIR = os.getcwd()+"/"
print(rainbow_print(" OPERATING OUT OF "+_DIR))
config = configparser.ConfigParser()
config.read(_DIR+"../config.ini")

# CONFIGURATION
monitoring = [
    "$spx", "$nvda", "$vix"
]
csv_header = [
    'text', 'sentiment', 'symbols', 'mentions', 'follower_count', 'following', 'screen_name', 'created_at'
]
banned_words = [
    "ANALYST PRICE", "CHAT", "COMMUNITY", "FOLLOW", "TARGET PRICE", "PRICE TARGET", "ALERTS", "DISCORD", "SIGNALS", "CHATROOM", "JOIN", "LINK", "TRADING COMMUNITY", "ALL THESE LEVELS", "CLICK HERE"
]
banned_accounts = [
    "LlcBillionaire", "atonesapple66", "stockmktgenius", "optionwaves", "CeoPurchases", "OptionsFlowLive", "TWAOptions", "Smith28301", "bishnuvardhan", "nappedonthebed", "TheTradingChamp", "SJManhattan", "MalibuInvest", "vaikunt305", "Sir_L3X", "bankston_janet", "Maria31562570", "LasherMarkus", "PearlPo21341371"
]
notification = "netgraph.wav"
csv_name = "spx_nvda_vix"
graph_interval = 1
save_interval = 20
post_tweet = False
show_plot = False
notification_on = True
analysis_tweet = """
📊🔮 #fintwit sentiment & speech similarity clustering
[ """+str(monitoring)+""" ]
"""

# This is the authentication for the Twitter API.
auth = tweepy.OAuthHandler(config['DEFAULT']['twitter_consumer'],
                           config['DEFAULT']['twitter_consumer_secret'])
auth.set_access_token(config['DEFAULT']['twitter_token'],
                      config['DEFAULT']['twitter_key'])

# Loading the pretrained model and tokenizer for financial sequences from HuggingFace.
tokenizer = AutoTokenizer.from_pretrained(
    "nickmuchi/deberta-v3-base-finetuned-finance-text-classification")
model = AutoModelForSequenceClassification.from_pretrained(
    "nickmuchi/deberta-v3-base-finetuned-finance-text-classification")

api = tweepy.API(auth)

DG = nx.DiGraph()

data = pd.DataFrame(columns=csv_header)


# class Network(nx.DiGraph):
#     def _temp():
#         print('')


class TweetPipeline(tweepy.Stream):

    def on_status(self, status):
        self.preprocess(status)

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def count_elements(self, json_objects):
        counts = Counter()
        for obj in json_objects:
            symbol_list = obj
            for symbol in symbol_list:
                counts[symbol] += 1
        return dict(counts)

    def sentence_similarity(self, sentences):
        """
        It takes a list of sentences, encodes them using the MiniLM model, and then returns the sentence
        embeddings

        :param sentences: a list of strings
        """
        tokenizer = AutoTokenizer.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2')
        encoded_input = tokenizer(sentences, padding=True,
                                  truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1])

    def infer_sentiment(self, input_text):
        """
        It takes in a string of text, encodes it into a sequence of integers, and returns the embedding
        vector for the last token in the sequence

        :param input_text: The text you want to embed
        :return: The last hidden state of the model.
        """
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model(input_ids)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def get_friends(self, screen_name):
        """
        It gets the first 10 pages of 200 friends for a given screen name

        :param screen_name: The screen name of the user you want to get the friends of
        :return: A list of strings.
        """
        followers = []
        for page in tweepy.Cursor(api.get_friends, screen_name=screen_name,
                                  count=200).pages(10):
            time.sleep(5)
            for user in page:
                name = f"{user.id} - {user.name} (@{user.screen_name})"
                followers.append(name)
        return followers

    def great_filter(self, tweet):
        """
        If the tweet is a retweet, an advertisement, or has no symbols, then it is not a great tweet

        :param tweet: a tweet object
        :return: A list of dictionaries.
        """
        if 'retweeted_status' in tweet.keys():
            print(termcolor.colored('⛔️ RT', 'red'))
            return False
        for word in banned_words:
            if word in tweet['text'].upper():
                print(termcolor.colored('⛔️ '+word, 'red'))
                return False
        for account in banned_accounts:
            if account.upper() == tweet['user']['screen_name'].upper():
                print(termcolor.colored('⛔️ '+account, 'red'))
                return False
        if len(tweet['entities']['symbols']) == 0 or len(tweet['entities']['symbols']) >= len(monitoring):
            print(termcolor.colored('⛔️ SYMBOLS', 'red'))
            return False
        else:
            return True

    def preprocess(self, tweet):
        """
        It takes a tweet, preprocesses it, and adds it to the graph

        :param tweet: the tweet object
        """
        _json = json.dumps(tweet._json, indent=2)
        tweet = json.loads(_json)
        if self.great_filter(tweet):
            screen_name = tweet['user']['screen_name']
            print("🧑‍💻 "+screen_name)
            _text = tweet['text']
            _mentions = tweet['entities']['user_mentions']
            _symbols = tweet['entities']['symbols']
            def _nolink(x): return re.sub(r"http\S+", "", x).replace("\n", "")
            print(termcolor.colored("🔮 "+_nolink(_text), 'green'))
            text = _nolink(_text)
            _sentiment = self.infer_sentiment(text)
            _created_at = tweet['created_at']
            _created_at = datetime.strptime(
                _created_at, "%a %b %d %H:%M:%S %z %Y")
            est_tz = pytz.timezone('America/New_York')
            est_dt = _created_at.astimezone(est_tz)
            created_at = est_dt.strftime("%a %b %d %H:%M:%S %Z %Y")

            sentiment = _sentiment.detach().numpy()[0].tolist()
            symbols = [x['text'].upper() for x in _symbols]
            mentions = [x['screen_name'] for x in _mentions]
            follower_count = tweet['user']['followers_count']
            # following = get_friends(tweet['user']['screen_name'])

            following = []

            data.loc[len(data)] = [text, sentiment, symbols,
                                   mentions, follower_count, following, screen_name, created_at]

            csv_data = [text, sentiment, symbols,
                        mentions, follower_count, following, screen_name, created_at]

            with open(_DIR+csv_name+".csv", 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(csv_header)
                writer.writerow(csv_data)
            node_id = len(data)
            DG.add_node(node_id, text=text, sentiment=sentiment, symbols=symbols,
                        mentions=mentions, follower_count=follower_count, following=following)
            for other_node_id, other_node_data in DG.nodes(data=True):
                if other_node_id == node_id:
                    pass
                else:
                    cosim = self.sentence_similarity([DG.nodes(data=True)[node_id]['text'], DG.nodes(
                        data=True)[other_node_id]['text']]).detach().numpy()[0][0]
                    for symbol in symbols:
                        if symbol in other_node_data["symbols"]:
                            DG.add_edge(other_node_id, node_id,
                                        weight=np.round(cosim, 4), alpha=np.round(cosim, 4))

            if len(DG.nodes()) % graph_interval == 0:
                self.plot_network()
            else:
                print("⏲️ "+termcolor.colored(str(len(DG.nodes())) +
                      "/"+str(graph_interval), "blue"))
        else:
            pass

    def plot_network(self):
        """
        It takes a tweet, preprocesses it, and then adds it to the graph.
        """
        plt.clf()  # clear the previous plot
        pos = nx.spring_layout(DG)
        labels = nx.get_node_attributes(DG, "symbols")
        sentiments = nx.get_node_attributes(DG, "sentiment")
        follower_count = nx.get_node_attributes(DG, "follower_count")
        label_values = [labels[node] for node in DG.nodes()]
        sentiment_values = [sentiments[node] for node in DG.nodes()]
        node_sizes = [follower_count[node] for node in DG.nodes()]
        pop_symbols = self.count_elements(label_values)
        colors = []
        for sentiment in sentiment_values:
            if (sentiment[0] > sentiment[1]) and (sentiment[0] > sentiment[2]):
                colors.append("pink")
            elif (sentiment[1] > sentiment[0]) and (sentiment[1] > sentiment[2]):
                colors.append("grey")
            elif (sentiment[2] > sentiment[1]) and (sentiment[2] > sentiment[1]):
                colors.append("lightgreen")
        edge_labels = nx.get_edge_attributes(DG, "weight")
        nx.draw_networkx_edge_labels(
            DG, pos, edge_labels, font_size=6, font_color="purple")
        nx.draw_networkx_labels(
            DG, pos, labels, font_size=8, font_color="black")
        nx.draw(DG, pos, with_labels=False, node_color=colors,
                node_size=node_sizes, edgecolors='purple', alpha=0.5)
        plt.title('Real-Time Network Analysis - FinTwit', fontsize=18)
        plt.figsize = (12, 8)
        if len(DG.nodes()) % save_interval == 0:
            dt = datetime.now().strftime("%m/%d/%Y-%H:%M:%S").replace("/", "_")
            _fn = _DIR+dt+'.png'
            plt.savefig(_fn, dpi=600)
            print(rainbow_print(" GRAPH SAVED TO "+dt+".png"))
            if notification_on:
                playsound(_DIR+notification)

        if post_tweet:
            analysis_tweet_media = api.media_upload(_fn)
            api.update_status(status=analysis_tweet, media_ids=[
                analysis_tweet_media.media_id])
        if show_plot:
            plt.ion()  # enable interactive mode
            plt.pause(0.1)  # pause for 0.1 seconds
            plt.show()


# Creating a stream of tweets
tweet_stream = TweetPipeline(
    auth.consumer_key, auth.consumer_secret, auth.access_token, auth.access_token_secret)
tweet_stream.filter(track=monitoring)
