# 10/1

I have created a text file of show dockets which I have found on a fan website. I cleaned up the file to make it easier to work with and removed irrelevant information.

I found a website containing Twitch Stream statistics going back 365 days. I contacted the owner of the website to obtain a csv file for the streamer Northernlion going back further than 365 days.

I joined the two files into a data frame and have been removing non-relevant information and making sure the rows for both files are lined up correctly.

I have created a developer account on Twitch and will use the API to develop a Python script that will give me a list of stream IDs for a given streamer. I will feed this list into a Python script I found which creates text files of Twitch comments if a video ID is provided.

# 10/7

I have added an MIT license to my project. I have been working on learning how openly I can use  the data that I've collected. The CSVs from Sullygnome are provided with permission and a request for attribution. The global emotes list was taken from the Twitch API. I have contacted the owner of the NLSS Docket website through his Reddit account and have recieved permission to use the data publicly.

I have written a worked in Jupyter to convert the global emotes JSON file to a text file of emote names seperated by newlines. I can use this to filter through comments on Twitch. For instance, if I'm looking up most frequent words, emotes are likely to be at the top of the list. I can use this emote list to filter them when looking for comments. I will look for a way to add the names of Northernlion channel specific emotes in addition to the global emotes. This may need to be added by hand, but there are many fewer channel specific emotes.

It seems the hardest part so far of aggrigating data lies in learning the Twitch API to get video IDs. Once that is worked out however, I don't forsee too much left in terms of collecting data. I created a seperate Twitch account called https://www.twitch.tv/nlss_project to aid in gathering data.

In retrying the Twitch comment downloading script I'm running into connection errors. I had it working before, so I will need to troubleshoot and learn whether the problem is on Twitch's side or mine.

# 10/8
Twitch changed their API and broke the Twitch comment downloader. I need to find a way to make it compatable with the new API or find a new app to use. ㅠ-ㅠ