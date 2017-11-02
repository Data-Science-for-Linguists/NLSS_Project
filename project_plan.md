# Updated Project Plan

A list of Video IDs has been compiled and fed to a comment downloading script.  I kept track of the videos which were split over multiple parts. The comment files generated from these videos will need to be put together. I have two files, one called VOD_ID and the other VOD_ID_Annotated. The difference is that the annotated file notes which videos need to be put together. I will attempt to put the comments together by comparing the video dates, as those that match are split streams.

I am working on setting up a data frame of the comments. It will contain a column for the date of the stream, one for the commenters, and one for the message. I will then use the date column to combine the data frame with the stream statistics data frame for the completed form. 

I have found a list of emotes used on Twitch. Twitch emotes work in that you type a word like Kappa and see a picture emote. I can use this list of emotes to filter in and out emotes from chat when doing my analysis.

From there my analysis of the data can begin. I would like to try a sentiment analysis task on chat. I have noticed that when games are introduced on stream, it often takes time for people to warm up to them. I have the statistics on what games are played on a stream and can see how the sentiment of chat changes as a game is played over the course of different streams. I can compare if the sentiment analysis matches with the stream stats on viewers per hour.

I'm curious if certain games, or crew members change the positivity of the chat's language. I could look at the positivity of chat by looking at the rates certain words occur at in different videos. This can than be graphed out to see if any games or crew members affect this statistic.

What are the most active users saying? I would like to look at the comments of users who post the most across all videos. Is it positive, negative, neutral? How do their comments compare to less active users? What percentage of chat is produced by 'power users'? I could try to build a machine learning model the predicts whether a user is an active user. I could also do the same sort of analysis while considering whether the user is a subscriber or not. The JSON file includes whether a user subscribes to the channel.

It also may be interesting to build a language model of Twitch Chat. 