# 10/1

I have created a text file of show dockets which I have found on a fan website. I cleaned up the file to make it easier to work with and removed irrelevant information.

I found a website containing Twitch Stream statistics going back 365 days. I contacted the owner of the website to obtain a csv file for the streamer Northernlion going back further than 365 days.

I joined the two files into a data frame and have been removing non-relevant information and making sure the rows for both files are lined up correctly.

I have created a developer account on Twitch and will use the API to develop a Python script that will give me a list of stream IDs for a given streamer. I will feed this list into a Python script I found which creates text files of Twitch comments if a video ID is provided.