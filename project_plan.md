# Updated Project Plan

A list of Video IDs has been compiled and working script takes these IDs to download the comments. This list is fed to the comment downloader, but it takes time to download them all and this is an ongoing process. I will continue to work on getting these comments all downloaded so that they can be added to the DataFrame. One thing I must keep track of is videos which have been split over multiple parts. The comment files generated from these videos will need to be put together. I have two files, one called VOD_ID and the other VOD_ID_Annotated. The difference is that the annotated file notes which videos need to be put together.

I have found a list of emotes used on Twitch. Twitch emotes work in that you type a word like Kappa and see a picture emote. I can use this list of emotes to filter in and out emotes from chat when doing my analysis.

From there my analysis of the data can begin. I would like to try a sentiment analysis task on chat. I have noticed that when games are introduced on stream, it often takes time for people to warm up to them. I have the statistics on what games are played on a stream and can see how the sentiment of chat changes as a game is played over the course of different streams. I can compare if the sentiment analysis matches with the stream stats on viewers per hour.

I'm curious if certain games, or crew members change the complexity of the chat's language. I could look at the complexity of chat through the length of words used. This can than be graphed out to see if any games or crew members affect this statistic.

What are the most active users saying? I would like to look at the comments of users who post the most across all videos. Is it positive, negative, neutral?