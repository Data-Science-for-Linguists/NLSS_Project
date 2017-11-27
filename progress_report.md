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

# 10/12
Pardon the language
https://clips.twitch.tv/MushySecretiveHeronTheTarFu

# 10/27
I've been experimenting with various new chat downloaders. Twitch-Chat-Downloader is updated somewhat regularly, but sometimes has trouble encoding characters, causing it to crash before finishing the full download. I've found one that seems promising called rechat-dl. It downloads the comments as a JSON. I'm trying to learn how to parse it so that I could retrieve just the comments from the file. I'm using a JSON Viewer to learn the format. I have been working on chat downloads from a folder that doesn't get pushed, but I will begin to push updates on a script to downloads chats.

I have finished a short script for downloading the JSON of comments and extracting a given comment from it. A copy of this called Download_Script has been added into a new folder called Pipeline. The Pipeline folder will show the basic steps of how file IDs are retrieved, fed into the comment downloader, and affixed to the full dataframe.

# 10/28
I haven't been able to find an elegant solution for how to get the VOD IDs. Instead, what I've started doing is opening all the links to the VODs, getting their ID from the URL and putting it in a text file. I can use this text file as the input for my VOD comment downloader.

I realized that some VODs are split into two parts. This means the comments will be over two files. I'm going to see if I can extract the date from the JSON files to group files together if they are from the same date/stream.

# 10/29
I've come across what appears to be missing VODs which will have to be accounted for when combining the comments with the dataframe. I'm having issues with the comment download script. When giving it a VOD ID it works fine, but when for-looping through a list of VOD IDs, it crashes.

# 10/30
I have started running my script that downloads all of the video comments. I had to use 'subprocess' in order to for loop through the video IDs and run the script. I am going to start working on a second list of video IDs which picks up where the current one left off. The first video list ends at a spot in which a few VODs were not uploaded. It should be ok if I don't have comments for every row in the dataframe, but I want to make sure that the comments are attached to the proper row instead of attaching to the missing VOD rows.

# 10/31
I found that the first object in every JSON file is information about the stream. Using the recorded_at attribute, I can pull the date of each VOD out of the comments and use this to attach VODs to comments and maybe to put together the comments split over multiple videos.

I encountered my first three part video while grabbing the last few video IDs.

# 11/1
All comment JSONs have been downloaded, and are now being organized to put into the full data frame.

I'm tring to figure out the best way to store the comments so that I can add them to the video data frame. I think a multi-index solution may be the best choice. My outer most index will be the date the video was recorded (which can help us put back together split VODs), the inner index could be the commenter, and then have a column for the comments.

# 11/17
It's been too long, let's get back into this. I have been working on the project exclusively on my desktop. However, all of the data that is used in my project is hosted on Github, so I have changed the NLSS_V2 and Comment_DataFrame files to read in the files in the Github hosted directories. This means I can easily switch back and forth between desktop and laptop.