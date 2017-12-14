# Final Report 

by Andrew Ryan

## 1. Background 

The Northernlion Live Super Show is a long running video game live stream hosted on the website Twitch.tv. As streamers play games, a live chat feed scrolls alongside the video. Unlike comments on video sites like Youtube, which allow an ex post facto conversation, the live nature of Twitch means that comments are a way to interact with streamers in real time. In some cases, streamers may even have [conversations](https://clips.twitch.tv/StormyVainEelCurseLit) with the collective audience. This project looks at the sentiment of Twitch comments and tries to predict the overall sentiment of comments in a stream based on viewership and episode information.

## 2. Data Collection 

A corpus was compiled consisting of 5,000,000 comments from 200 episodes of the Northernlion Live Super Show. Corpus creation started with manually compiling a list of unique [video identifiers](Pipeline/VOD_ID_full.txt) found at the end of each video's URL. These video ids were fed into [rechat-dl](https://github.com/KunaiFire/rechat-dl) which downloads a JSON file of comments for each video. Information regarding the date of the comment, the username, and the content of the message were compiled into a dataframe of comments. Another dataframe was created from information about the [docket](http://twoandahalfscums.blogspot.co.uk/p/nlss.html) (games played on a stream, the users playing the games), and [viewership statistics](sullygnome.com/channel/Northernlion). These two dataframes were put together for a full dataframe which contained entries for each of the episodes with columns for the docket, stream statistics, and comments.

## 3. Data Cleaning 

Throughout each step of creating the corpus data had to be cut. The original list of dockets contained entries for 588 shows, while the files containing stream statistics only contained information on 441 shows. Removing duplicate shows and unrelated streams left a list of 303 videos that could be looked at. However, after downloading JSON files of comments for each of the videos, it was discovered that Twitch did not retain chat record chat records before a certain date. Removing files that did not contain comments resulted in 200 videos that could be analyzed.

The docket and player information were turned into lists. Information about the crew of players were standardized by converting various nicknames into a uniform name. Several show dates had been entered wrong in the docket list. These episodes were checked against records such as Tweets that mentioned the episode in order to fix the dates.

## 4. Sentiment Analysis 

Sentiment Analysis was performed with [VADER](https://github.com/cjhutto/vaderSentiment). VADER was chosen as it is "specifically attuned to sentiments expressed in social media." VADER is a lexically based sentiment analyzer. A dictionary of words and their sentiment scores are created by having human raters from Amazon Mechanical Turk judge each word. When VADER analyzes a sentence, it first analyzes individual words (and emoticons) and assigns them a sentiment score between -4 (more negative) and 4 (more positive). The sentiment of a sentence is the sum of the scores of each word in the sentence normalized between -1 (more negative) and 1 (more positive).

In addition to the base word scores, VADER's sentiment scores factor in features such as capitalization, punctuation, negation, and degree modifiers. These scores, like the lexical scores, are based on human ratings from MTurk.

## 5. Score Prediction 

