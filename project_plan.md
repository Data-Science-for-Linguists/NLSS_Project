I would like to perform a linguistic analysis on the livestream, "Northernlion Live Super Show." 

I have collected a list of show dockets spanning from August 2017 to February 2013. Each docket contains two lines. The first line contains the show's date and the members who appeared in the episode. The second line is a list of the games played in each section of the show. I have cleaned up the document a bit, but it still needs a bit of work (e.g. remove "w/" and replace with ",").

This list is paired with a csv of stream statistics of 441 streams hosted by the streamer. The information from the two files is paired by date, keeping only the stats of shows that the docket list tells us are NLSS episodes.

A program will be written using the Twitch API to gather a list of Twitch video IDs from a streamer. This list will be fed to a script which downloads a text file of comments for a Twitch video. These text files will be paired with the data frame containing stream info.

Specific questions I would like to address in this project include:

* How many times can a game be played before viewership declines?
* Does chat respond more positively or negatively to new games?
* What is the ratio of unique messages to copypasta?
* Which games lead to more complex sentences and which games lead to simpler sentences?