# Final Report 

by Andrew Ryan

## 1. Background 

The Northernlion Live Super Show is a long running video game live stream hosted on the website Twitch.tv. As streamers play games, a live chat feed scrolls alongside the video. Unlike comments on video sites like Youtube, which allow an ex post facto conversation, the live nature of Twitch means that comments are a way to interact with streamers in real time. In some cases, streamers may even have [conversations](https://clips.twitch.tv/StormyVainEelCurseLit) with the collective audience. This project looks at the sentiment of Twitch comments and trys to predict the overall sentiment of comments in a stream based on viewership and episode information.

## 2. Data Colection 

A corpus was compiled consisting of 5,000,000 comments from 200 episodes of the Northernlion Live Super Show. Corpus creation started with manually compiling a list of unique [video identifiers](Pipeline/VOD_ID_full.txt) found at the end of each video's URL. These video ids were fed into [rechat-dl](https://github.com/KunaiFire/rechat-dl) which downloads a JSON file of comments for each video. Information regarding the date of the comment, the username, and the content of the message were compiled into a [dataframe](NLSS_V3.ipynb#Comment-Dataframe)

## 3. Data Cleaning 

## 4. Sentiment Analysis 

## 5. Score Predicition 

