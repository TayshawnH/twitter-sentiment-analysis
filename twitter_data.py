import re
import tweepy
import os
from dotenv import load_dotenv

load_dotenv()


class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''

    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = os.getenv("API_KEY")
        consumer_secret = os.getenv("API_SECRET")
        access_token = os.getenv("API_TOKEN")
        access_token_secret = os.getenv("API_TOKEN_SECRET")

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweets(self, query, count):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets
            fetched_tweets = tweepy.Cursor(self.api.search, q=query,
                                           include_entities=True,
                                           tweet_mode='extended',
                                           since="2021-01-01",
                                           lang="en").items(count)

            # parsing tweets one by one
            for tweet in fetched_tweets:
                print(tweet.retweet_count)
                parsed_tweet = self.clean_tweet(tweet.full_text)
                # print(tweet.retweet_count)
                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            # return parsed tweets
            return tweets

        except tweepy.TweepError as e:
            print("Error : " + str(e))


def main():
    api = TwitterClient()
    tweets = api.get_tweets(query="league of legends", count=1500)
    # print(tweets)
    for tweet in enumerate(tweets):
        # print(idx, tweet["text"])
        with open('game_data/League.csv', 'a', newline='') as f:
            f.write("%s\n" % tweet[1])


if __name__ == "__main__":
    main()
