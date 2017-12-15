
# UPDATE

This notebook is no longer being used. Please look at the most recent version, NLSS_V2 found in the same directory.

My project looks at the Northernlion Live Super Show, a thrice a week Twitch stream which has been running since 2013. Unlike a video service like Youtube, the live nature of Twitch allows for a more conversational 'live comment stream' to accompany a video. My goal is to gather statistics about the episodes and pair this with a list of all the comments corrosponding to a video. Using this I will attempt to recognize patterns in Twitch comments based on the video statistics.


```python
# Every returned Out[] is displayed, not just the last one. 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


```python
import nltk
import pandas as pd
import numpy as np
```

A text file containing basic information about every NLSS episode must be organized into something usable


```python
with open(r'data\NLSS_Dockets.txt') as f:
    file = f.read()
shows = file.split('\n\n') #split into every show
shows[:5]
```




    ['(August 24, 2017) Nick View (NL, RLS, CS, rob)\nPasspartout, Party Panic, Pinturillo',
     '(August 23, 2017) Nick View (NL, RLS, rob w/ Baer, LGW, HCJ)\nAbsolver, Golf It, Quiplash',
     '(August 21, 2017) Nick View (NL, RLS, JS, rob)\nFire Pro Wrestling World, Ultimate Chicken Horse, Blood Party',
     '(August 17, 2017) Nick View (NL w/ Sin, RLS, LGW, HCJ, Baer)\nGeoguessr, Golf It, Quiplash',
     '(August 16, 2017) Nick View (NL, RLS w/ rob, Baer, LGW, Dan)\nNidhogg 2, Speedrunners, Pinturillo']



This text file was taken from a webpage and so it contains links to Nick's livestream. Let's get rid of this since it's not needed.


```python
index = 0
for s in shows:
    shows[index] = s.replace(' Nick View', '')
    index+=1
shows[-10:]
```




    ['(March 27, 2013) (NL, RLS, JS w/ Ohm)\nDark Souls Invasions, Trivia, Arma III, Ask me anything on Twitter',
     '(March 18, 2013) (NL, RLS, JS)\nDark Souls Invasions, Trivia, Rollercoaster Tycoon, More Dark Souls Invasions, Ask me anything on Twitter',
     '(March 14, 2013) (NL, RLS, JS w/ MALF)\nDark Souls Invasions, Trivia, Worms Revolution, Ask me anything on Twitter',
     '(March 13, 2013) (NL, RLS, JS w/ Ohm)\nDark Souls Invasions, Trivia, The Showdown Effect',
     '(March 11, 2013) (NL, RLS, JS w/ Ohm)\nDark Souls Invasions, Trivia, More Dark Souls Invasions, Arma III, Ask me anything on Twitter',
     '(March 6, 2013) (NL, RLS, JS)\nDark Souls invasions, Trivia, Tomb Raider, Ask me anything on Twitter',
     "(March 4, 2013) (NL, RLS, JS)\nDelver's Drop with Ryan Baker and Ryan Burrell, Dark Souls, Trivia, More Dark Souls, Ask me anything on Twitter",
     '(February 28, 2013) (NL, RLS, JS)\nDark Souls, Trivia, Trials Evolution, Ask me anything on Twitter',
     '(February 27, 2013) (NL, Kate)\nDark Souls, Trivia, More Dark Souls, Ask me anything on Twitter',
     '(February 25, 2013) (NL)\nRunner 2, Super House of Dead Ninjas, Trivia, Hitman: Bloodmoney, Ask me anything on Twitter\n']



Now I need to split up each show into their meaningful parts. Let's start with the games played on each episode.


```python
games = []
for s in shows:
    g = s.split('\n') #Text files has games on second line
    games.append(g[1])
games
```




    ['Passpartout, Party Panic, Pinturillo',
     'Absolver, Golf It, Quiplash',
     'Fire Pro Wrestling World, Ultimate Chicken Horse, Blood Party',
     'Geoguessr, Golf It, Quiplash',
     'Nidhogg 2, Speedrunners, Pinturillo',
     'Hitman, Ben & Edd Blood Party, London 2012',
     'Ultimate Chicken Horse, Tower Unite, Who Wants To Be A Millionaire, Super Hexagon',
     'Passpartout, Witch It, Quiplash',
     'Afterbirth+, Golf with your Friends, Guesspionage',
     'Tricky Towers, Sonic and All Stars Racing Transformed, London 2012',
     'Ben and Ed - Blood Party, Ball 3D: Soccer Online, Quiplash',
     'Fortnite, Golf It, Gang Beasts',
     'Cannon Brawl, Goldeneye: Source, Ball 3D: Soccer Online',
     'Hitman, Ben & Ed - Blood Party, Tower Unite',
     'Ben and Ed - Blood Party, Golf with Your Friends, Pinturillo',
     "Ben and Edd's Blood Party, London 2012, Quiplash",
     'Passpartout, Ultimate Chicken Horse, Tower Unite',
     'The End Is Nigh, Passpartout, Quiplash',
     'Hitman, (continued), Tower Unite, Guesspionage',
     'Passpartout: The Starving Artist, Duck Game, Golf It',
     'Afterbirth+, Sonic All Stars Racing Transformed, Pinturillo',
     'Passpartout: The Starving Artist, Redout, Quiplash',
     'Passpartout, Ultimate Chicken Horse, Guesspionage',
     'Bloody Trapland 2, Passpartout: The Starving Artist, London 2012',
     'Hitman, Witch It, Family Feud',
     'Passpartout, Who Wants to be a Millionaire?, Goldeneye Source',
     'Passpartout, Battlegrounds, London 2012',
     'Golf It, Golf With Your Friends, Vertiginous Golf',
     'Afterbirth+, Gang Beasts, Goldeneye Source, Golf With Your Friends',
     'Monolith, Ultimate Chicken Horse, Quiplash',
     'Hitman, Speedrunners, Goldeneye Source',
     'Dead Cells, Duck Game, Family Feud Decades',
     'Battlegrounds, Golf With Your Friends, Quiplash',
     'Hitman, Helldivers, Goldeneye Source',
     'Dead Cells, Ultimate Chicken Horse, Pinturillo',
     'Afterbirth+, Goldeneye Source, Golf it, Golf With (Your) Friends',
     'Hitman, Half-Dead, Worms Revolution',
     'Dead Cells, Quake Champions, Quiplash',
     'Dead Cells, Ultimate Chicken Horse, Golf With (Your) Friends',
     'Hitman, WWE 2k16, Quiplash',
     'Afterbirth+, Lethal League, Quiplash',
     'Afterbirth+, Golf With Your Friends, London 2012',
     'Strafe, Hitman, Family Feud Decades',
     'Afterbirth+, Sonic All Stars Racing Transformed, Quiplash',
     'Tumbleseed, Ultimate Chicken Horse, Family Feud Decades',
     'Hitman, Golf With Your Friends, Quiplash',
     'SpyParty, Family Feud (Nick plays Shadow Warrior 2), Pinturillo',
     'Golf With Your Friends, Gang Beasts, Quiplash',
     'Ultimate Chicken Horse, Who Wants To Be a Millionaire, London 2012',
     'Afterbirth+, Battlegrounds, Pinturillo',
     'Afterbirth+, Golf It, Quiplash',
     'Afterbirth+, Battlegrounds, Pinturillo',
     'Afterbirth+, Golf With (Your) Friends, Guesspionage',
     'Hitman Roulette, Battlegrounds, Quiplash',
     'Afterbirth+, Battlegrounds, Sonic All Stars Racing Transformed',
     'Ultimate Chicken Horse, Move or Die, RedOut',
     "Afterbirth+, Hitman Roulette, PlayerUnknown's Battlegrounds",
     'Afterbirth+, Nick plays Meatboy, Golf With (Your) Friends, Duck Game',
     'Afterbirth+, We Need to Go Deeper, Pinturillo',
     'Afterbirth+, Streets of Rogue, Quiplash',
     'Afterbirth+, Disc Jam, Pinturillo',
     'Afterbirth+, Golf It, Tricky Towers',
     'Afterbirth+, Gang Beasts, Pinturillo',
     'Afterbirth+, Sonic All Stars Racing Transformed, Quiplash',
     'Afterbirth+, Quiplash, London 2012',
     'Afterbirth+, Ultimate Chicken Horse, Sonic All Stars Racing Transformed',
     'Afterbirth+, Disc Jam, Pinturillo',
     'Afterbirth+, Golf It, For Honor',
     'Afterbirth+, Duck Game, Quiplash',
     'Afterbirth+, NLSS mod, Quiplash, Pinturillo',
     'Ultimate Chicken Horse, Golf It, For Honor',
     'Afterbirth+, Scribblenauts, Invisigun Heroes',
     'Afterbirth+, For Honor, Quiplash',
     'Afterbirth+, Sonic All Star Racing Transformed, (continued), Golf With Your Friends',
     'Afterbirth+, Disc Jam, Gang Beasts',
     'Afterbirth+, Sonic All Stars Racing Transformed, Pinturillo',
     'Afterbirth+, Ultimate Chicken Horse, Golf With (Your) Friends',
     'Afterbirth+, Scribblenauts, Quiplash',
     'Enter the Gungeon, Disc Jam, Pinturillo',
     'Afterbirth+, Ultimate Chicken Horse, Quiplash',
     'Afterbirth+, Scribblenauts attempt, Who Wants To Be A Millionaire, Space Beast Terror Fright, (continued)',
     'Afterbirth+, Who Wants to be a Millionaire, Golf With (Your) Friends',
     'Afterbirth+, Ultimate Chicken Horse, Pinturillo',
     'Afterbirth+, Disc Jam, Trivia Murder Party, Guesspionage, Quiplash Continued',
     'Afterbirth+, Dead By Daylight, Quiplash',
     'Afterbirth+, Disc Jam, Gang Beasts',
     'Afterbirth+ (Josh name mod), Disc Jam, London 2012',
     'Afterbirth+, Gang Beasts, Quiplash',
     'Afterbirth+, RedOut (continued), Pinturillo',
     'Antibirth, Hearthstone, Space Beast Terror Fright, Duck Game',
     'Antibirth, Hearthstone, Ultimate Chicken Horse, Quiplash',
     'Antibirth, Hearthstone, Guesspionage, Pinturillo',
     'Antibirth, Gang Beasts, Quiplash',
     'Antibirth, Gang Beasts, H1Z1 King of the Kill',
     'Antibirth, Friday the 13th',
     'Ultimate Chicken Horse, Astroneer, H1Z1 King of the Kill',
     'Ultimate Chicken Horse, H1Z1 King of the Kill,Quiplash',
     'Afterbirth, Gang Beasts, Dead by Daylight',
     'Tricky Towers, Ultimate Chicken Horse, Duck Game',
     'Ultimate Chicken Horse, Speedrunners, Pinturillo',
     'Afterbirth, Tumble Seed, Tricky Towers, London 2012',
     'Ultimate Chicken Horse, Move or Die, Quiplash',
     'Ultimate Chicken Horse, Bombernauts, Golf With (Your) Friends',
     'Ultimate Chicken Horse, Gang Beasts, Duck Games',
     'Tricky Towers, Dead By Daylight, Quiplash',
     'Ultimate Chicken Horse, London 2012, Keep Talking and Nobody Explodes',
     'Afterbirth, Guesspionage, Pinturillo',
     'Ultimate Chicken Horse, Gang Beasts, Golf With (Your) Friends',
     'Ultimate Chicken Horse, Runbow, Quiplash 2',
     'Afterbirth, Worms WMD, Golf With (Your) Friends',
     'Ultimate Chicken Horse, Quiplash, Rocket League',
     'Ultimate Chicken Horse, The Culling, Pinturillo',
     'Ultimate Chicken Horse, London 2012, Quiplash',
     'Ultimate Chicken Horse, Gang Beasts, Duck Game',
     'Ultimate Chicken Horse, Goofball Goals, Gang Beasts',
     'Afterbirth, Speedrunners, Quiplash',
     'Ultimate Chicken Horse, Golf With Your Friends, The Culling',
     'Afterbirth, Tricky Towers, Pinturillo, London 2012',
     'Ultimate Chicken Horse, Dead By Daylight attempt, Guesspionage, Quiplash 2',
     "Ultimate Chicken Horse, Who's Your Daddy, Murder Trivia Party, Keep Talking and No One Explodes",
     'Afterbirth, Guesspionage, Trivia Murder Party, Golf With Your Friends',
     'Ultimate Chicken Horse, Trivia Murder Party, Pinturillo',
     'Ultimate Chicken Horse, Guesspionage, Quiplash 2, More Guesspionage',
     'Afterbirth, SpyParty, Worms W.M.D.',
     'Ultimate Chicken Horse, Brawlhalla, Pinturillo, Quiplash',
     'Shadow Warrior 2, Golf with Your Friends, Ultimate Chicken Horse',
     'Ultimate Chicken Horse, London 2012, Pinturillo',
     'Tricky Towers, Ultimate Chicken Horse, Quiplash',
     'Afterbirth, Viking Squad, Ultimate Chicken Horse',
     'Ultimate Chicken Horse, Tricky Towers, ShellShock Live',
     'Afterbirth, Duck Game Attempt, Pinturillo, Golf With (Your) Friends',
     'Afterbirth, Ultimate Chicken Horse, Tricky Towers',
     'Afterbirth, Golf With (Your) Friends, Quiplash',
     'Afterbirth, London 2012, Pinturillo',
     'Afterbirth, Rocket League',
     'Afterbirth, Dead By Daylight, Rocket League',
     'Geoguessr, Quiplash, Golf With Friends',
     'Afterbirth, Worms WMD, Pinturillo',
     'Half Dead, WWE 2K16, London 2012',
     'Afterbirth, Goldeneye Source, Quiplash',
     'GeoGuessr, Monsters and Monocles, London 2012',
     'Afterbirth, GeoGuessr, Quiplash',
     'Afterbirth, Worms WMD, London 2012',
     'Afterbirth, Death Stair, Pinturillo',
     'Afterbirth, Dead By Daylight, Golf With Friends, (continued)',
     'Afterbirth, Worms W.M.D., Pinturillo',
     'Afterbirth, London 2012',
     'Afterbirth, Quiplash, London 2012',
     'Afterbirth, Dead by Daylight (continued) Pinturillo',
     "Afterbirth, Who's Your Daddy, London 2012 with echoes",
     'Afterbirth, WWE 2K16, London 2012',
     'Afterbirth, London 2012, The Culling',
     'GeoGuessr, RimWorld, Quadrilateral Cowboy',
     'Afterbirth, Evolve, Videoball',
     'Afterbirth, Brawlhalla, Pinturillo',
     'Afterbirth, WWE 2K16, Golf With Friends',
     'Afterbirth, WWE 2K16, Videoball',
     'Afterbirth, Worms Revolution, Videoball',
     'Afterbirth, Necropolis, Pinturillo 2',
     'Afterbirth, WWE 2K16',
     'Afterbirth, The Culling, Pinturillo',
     'Afterbirth, WWE 2K16, Dead By Daylight',
     'Afterbirth, Quiplash, Overwatch',
     'Afterbirth, Duck Game, Pinturillo',
     'Afterbirth, (aborted attempt to play Gang Beasts), Superfight, Golf With Friends',
     'Afterbirth, Dead By Daylight, Pinturillo',
     'Afterbirth, Dead By Daylight, Drawful 2',
     'Afterbirth, Quiplash, Pinturillo',
     'Afterbirth, Duck Game, Dead by Daylight',
     'Afterbirth, Golf with Friends, Family Feud Decades',
     'Afterbirth, Goofball Goals, ShellShock Live',
     'Gungeon, Table Top Racers: World Tour, Pintrullo',
     'Afterbirth, Golf With Friends, Overwatch',
     'Afterbirth, Duck Game, Overwatch',
     'Afterbirth, Gang Beasts, Quiplash, Pinturillo',
     'Afterbirth, Gang Beasts, The Culling, Overwatch',
     'Geoguessr, The Culling, Gang Beasts',
     'Geoguessr, The Culling, Gang Beasts',
     "Afterbirth, Who's Your Daddy, Golf With Friends, Quiplash",
     'Gungeon, Rocket League, Golf with Friends',
     'Afterbirth, Gang Beasts, Pinturillo',
     'Enter the Gungeon, The Culling, Overwatch',
     'Afterbirth, Gang Beasts, Gang Beasts',
     'Enter the Gungeon, Moonshot, The Culling',
     'Enter the Gungeon, Knight Squad, Duck Game',
     'Enter the Gungeon, GeoGuessr, The Culling, Atlas Reactor',
     'Enter the Gungeon, The Culling',
     'Dark Souls 3, Dark Souls 3, Dark Souls 3',
     "Afterbirth, Who's Your Daddy, Quiplash",
     'Enter the Gungeon, The Culling, Shellshock Live',
     'Dark Souls 3, Dark Souls 3, Dark Souls 3',
     'Enter the Gungeon, Lethal League, Golf with Friends',
     'Enter the Gungeon, The Culling, Shellshock Live',
     'Afterbirth, Duck Game, Pinturillo',
     'Afterbirth, The Culling, Golf With Friends',
     'Afterbirth, Shellshock Live, Pinturillo 2',
     'Afterbirth, Space Food Truck, Shellshock Live!',
     'Afterbirth, The Culling, (continued), Golf with Friends',
     'Afterbirth, Enter the Gungeon, (continued), Shellshock Live',
     'Afterbirth, Quiplash',
     'Afterbirth, Mount Your Friends with MALF, Worms Revolution with MALF',
     'Afterbirth, Half Dead, Golf With Friends with rob',
     'Afterbirth, Obliteracers, Golf with Friends',
     'Afterbirth, Atlas Reactor, Goofball Goals',
     'Afterbirth, Duck Game, Golf With Friends and Family Feud',
     'Afterbirth, Toribash, Golf With Friends, Rock of Ages',
     'Afterbirth, The Political Machine 2016, American Truck Sim',
     'Afterbirth, Golf With Friends with MALF, The Forest with MALF',
     'Afterbirth, Golf With Friends, Move or Die',
     'Afterbirth, Golf with Friends, Duck Game, Cobalt',
     'Afterbirth, SpyParty, The Forest with MALF',
     'Afterbirth, Cobalt, Quiplash (Nick plays Super Meat Boy)',
     'Afterbirth, Secret Ponchos, Family Feud Decades',
     'Afterbirth, Family Feud Decades (Nick plays Afterbirth), The Forest',
     "Afterbirth, Goofball Goals, Spelunky, Who's Your Daddy?",
     'Afterbirth, BroForce, failed attempt to play Worms Revolution, Duck Game',
     'Afterbirth, Move or Die!, Worms Revolution',
     'Afterbirth, Spy Party, The Forest',
     'Afterbirth, Duck Game, Vertiginous Golf',
     'Afterbirth, Cryptark, Bombernauts with rob',
     'Afterbirth, Duck Game with rob, Quiplash with rob',
     'Afterbirth, Tharsis, Duck Game',
     'Afterbirth, Rocket League, Quiplash with alpacapatrol, Nick plays Super Meat Boy',
     'Afterbirth, Duck Game, Helldivers',
     "Afterbirth, Who's Your Daddy, Duck Game with alpacapatrol",
     'Afterbirth, Deathstate, Worms Revolution',
     'Afterbirth, Duck Game, Rainbow Six Siege',
     'Afterbirth, Geoguessr, Rainbow Six Siege',
     'Afterbirth, Duck Game, Rocket League Ice Hockey',
     'Afterbirth, Duck Game, Rainbow Six Siege',
     'Afterbirth, Rainbow Six Siege',
     'Rebirth, GeoGuessr, Rainbow Six Siege',
     'Afterbirth, Spelunky, Super Meat Boy, Worms Revolution',
     'Afterbirth, Duck Game, Helldivers',
     'Afterbirth, Dark Souls Race (continued), 20XX',
     'Afterbirth, Duck Game, More Duck Game',
     'Afterbirth, Keep Talking and No Body Explodes (continued), Rocket League',
     'Afterbirth, Dungeon of the Endless, XCom: Enemy Within',
     'Afterbirth, Duck Game',
     'Afterbirth, Duck Game',
     'Afterbirth, Spelunky',
     'Afterbirth, Worms Revolution (continued), Tabletop Simulator: Superfight',
     'Afterbirth, Rocket League, Duck Game',
     'Afterbirth, Speedrunners, Brawlhalla',
     'Afterbirth, Duck Game, Quiplash, Afterbirth',
     'Afterbirth, Duck Game, Rocket League',
     'Afterbirth, Duck Game, Rocket League',
     'Afterbirth, Afterbirth, Afterbirth',
     'Afterbirth, (continued), Duck Game',
     'Rebirth, Downwell, Cannon Brawl',
     'Dark Souls, Dark Souls, Dark Souls.',
     'Spelunky, Keep Talking and Nobody Explodes, Rocket League',
     'Spelunky, Nuclear Throne, Hard West, Geoguessr',
     'Downwell, Nuclear Throne (continued), GeoGuessr',
     'Spelunky, Keep Talking and Nobody Explodes (Continued), Jackbox Party Pack 2',
     'Rebirth, Keep Talking and Nobody Explodes (continued), Rocket League',
     'Rebirth, Keep Talking and No One Explodes (continued), Rocket League',
     'Rebirth, Worms Revolution, Jackbox Party Pack 2 (Nick plays Super Meat Boy)',
     'Rebirth, Duck Game (continued), Keep Talking And Nobody Explodes!',
     'Rebirth, Rampage Knights (continued), Bombernauts',
     'Rebirth, Nuclear Throne, Duck Game (continued), Are you smarter than a fifth grader?',
     'Rebirth, TowerClimb (continued), Rocket League',
     'Rebirth, Goofball Goals, 20XX',
     'Rebirth, Quiplash/Super Meat Boy, Worms Revolution',
     'Rebirth, Speedrunners, Spelunky',
     'Rebirth, Shovel Knight DLC (continued), 20XX',
     'Rebirth, Nuclear Throne, Worms Revolution (continued), Family Feud 2010',
     'Rebirth, Nuclear Throne, Party Hard (continued), Tabletop Simulator: Superfight',
     'Rebirth, Dandy, Rampage Knights',
     'Rebirth, Duck Game (continued), Tabletop Simulator: Superfight',
     'Rebirth, Bro Force (continued), Tabletop Simulator: Coup, Tabletop Simulator: Superfight',
     'Rebirth, Turbo Dismount (continued), Family Feud (Nick drew Donald Trump)',
     'Rebirth, Full Mojo Rampage, Worms Revolution (continued), Duck Game',
     'Rebirth, Rocket League, Vertiginous Golf',
     'Rebirth, Goofball Goals (continued), SpyParty',
     'Rebirth, Worms Revolution (continued), Bombernauts',
     'Rebirth, Nuclear Throne, Duck Game (continued), Game Dev Tycoon',
     'Rebirth, Nuclear Throne (continued), Who Wants to Be a Millionaire?',
     'Rebirth Nuclear Throne, Super Mutant Alien Assault (continued), Worms Revolution',
     'Rebirth, Nuclear Throne, OlliOlli 2: Olliwood, Rocket League',
     'Rebirth, Olympia Rising, Quiplash',
     'Rebirth, Brawlhalla (continued), Rocket League',
     'Rebirth, Bombernauts (continued), Duck Game',
     'Rebirth, Duck Game (continued), Speedrunners',
     'Rebirth & audio problems, (continued), Rocket League, Nuclear Throne, Bombernauts (continued)',
     'Rebirth, Nuclear Throne, Family Feud 2010 (continued), Rocket League',
     'Rebirth, Guild of Dungeoneering (continued), Rocket League',
     'Rebirth, Duck Game (continued), Rocket League versus Lirik',
     'Rebirth, Quiplash (continued), Rocket League',
     'Rebirth, Nuclear Throne, Cannon Brawl, Rocket League (continued), Duck Game, More Rocket League',
     'Rebirth, Quiplash (Continued), Rocket League',
     'Rebirth, Rocket League, Vertiginous Golf (continued), Duck Game',
     'Rebirth, Duck Game (continued), Rocket League',
     'Rebirth, failed attempt at Duck Game, SpeedRunners, Rocket League',
     'Rebirth, Shovel Knight Race (continued), Family Feud!',
     'Rebirth, Avalanche 2: Super Avalanche, Brawlhalla',
     'Rebirth, Speedrunners, Duck Game',
     'Rebirth, Titan Souls Race (continued), Game Dev Tycoon',
     'Rebirth, Nuclear Throne, Enter the Gungeon (continued), Vertiginous Golf',
     'Rebirth, Nuclear Throne, Enter the Gungeon (continued), SpyParty',
     'Rebirth, Nuclear Throne, Family Feud (continued), Duck Game',
     'Rebirth, Dirty Bomb (continued), Trivia, Duck Game',
     'Rebirth, Nuclear Throne, Who Wants to be a Millionaire (continued), Duck Game',
     "Rebirth, Nuclear Throne, Big Pharma (continued), Trivia's Triumphant Return, Toy Box Turbo",
     'Nuclear Throne, Home Improvisation (continued), Duck Game',
     'Rebirth, Ronin (continued), Captain Forever Remix',
     'Magicka 2, more Magicka 2, even more Magicka 2',
     'Rebirth, Nuclear Throne, Metagame in Tabletop Simulator (continued), Action Henk',
     'Rebirth, Nuclear Throne, Catacomb Kids (continued), Action Henk',
     'Rebirth, Nuclear Throne, Goofball Goals, Brawlhalla',
     'Rebirth, SpyParty, Vertiginous Golf',
     'Rebirth, SpyParty, Brawlhalla',
     'Rebirth, Vertiginous Golf, Spelunky, Nuclear Throne',
     'Rebirth, Nuclear Throne, Who wants to be a millionaire, Google Feud (continued), attempt to play Armello, Brawlhalla',
     'Rebirth, Goofball Goals (continued) Speedrunners',
     'Rebirth, Captain Forever Remix (continued), Brawlhalla with rob, Baer',
     'Rebirth, Nuclear Throne, Convoy',
     'Rebirth, Convoy, Lethal League, Brawlhalla',
     'Rebirth, Goofball Goals (continued), Square Heroes',
     'Rebirth, Nuclear Throne, Killing Floor 2',
     'Rebirth, Killing Floor 2 (continued)',
     'Rebirth, Catacomb Kids, Brawlhalla',
     'Rebirth, Mount Your Friends with Baer, rob (continued) Square Heroes',
     'Rebirth, Speedrunners with rob, Nuclear Throne & twitch problems',
     'Rebirth, Goofball Goals (continued), Cannon Brawl',
     'Rebirth, Captain Forever Remix (continued), Google Feud with Baer',
     'Rebirth, Catacomb kids (continued), Brawlhalla',
     'Rebirth, Captain Forever Remix (continued), Brawlhalla with rob, Baer',
     'Rebirth, Goofball Goals, Spelunky, Nuclear Throne',
     'Rebirth, Nuclear Throne (continued), Google Feud',
     'Rebirth, Family Feud Decades (continued), Brawlhalla with rob, Baer',
     'Rebirth, Goofball Goals (continued), Catacomb Kids',
     'Rebirth, Geoguessr (continued), Drawful',
     'Rebirth, Move or Die with rob, Baer (Continued), Brawlhalla with rob',
     'Rebirth, Catacomb Kids, Speedrunners with rob, fox',
     'Rebirth, Toribash, Worms: Revolution',
     'Rebirth (continued), Catacomb Kids, Family Feud Decades',
     'Rebirth, SpyParty, Mount your Friends',
     'Rebirth, Goofball Goals, Speed Runners',
     'Rebirth, Family Feud, Who wants to be a millionaire (continued), Robot Roller Derby Disco Dodgeball',
     'Rebirth, Goofball Goals (continued), Armello',
     'Rebirth, Geoguessr (continued), Frozen Cortex with Baer',
     'Rebirth, SpyParty, (continued), Spelunky Mount your friends',
     'Rebirth, Family Feud Decades (continued), Darkest Dungeon (continued)',
     'Rebirth, Family Feud Decades with rob, Cannon Brawl',
     'Rebirth, Goofball Goals (continued), Speedrunners with rob',
     'Rebirth, Dying Light co-op (continued), Nuclear Throne',
     'Rebirth, Fibbage, Spelunky',
     'Rebirth, Goofball Goals (continued), Lethal League',
     'Rebirth, Drawful (continued), Speedrunners',
     'Rebirth, Goofball Goals (continued), Cannon Brawl',
     'Rebirth, Goofball Goals (continued), Spelunky, Mount Your Friends',
     'Rebirth, Fibbage Continued, Battleblock Theater',
     'Rebirth, Goofball Goals, Spelunky',
     'Rebirth, Geoguessr (continued), Fibbage',
     'Rebirth, Goofball Goals (continued), Mount Your Friends',
     'Rebirth, Canon Brawl (continued), Fibbage (The Jackbox Party Box) with Kate, Baer',
     'Rebirth, Goofball Goals (Continued), Distance',
     'Rebirth, Goofball Goals (continued), Lethal League',
     'Rebirth, Goofball Goals (continued), Mount Your Friends',
     'Rebirth, Goofball Goals, (continued), I am Bread',
     'Rebirth, GeoGuessr, (Continued), Dungeon of the Endless, Re-rebirth',
     'Rebirth, GeoGuessr (Continued), Spelunky, Re-rebirth',
     'Binding of Isaac: Rebirth, Goofball Goals Continued, Vertiginous Golf',
     'Binding of Isaac: Rebirth, Goofball Goals, (Continued), SpyParty',
     'Binding of Isaac: Rebirth, Mount your Friends (continued), Worms Revolution',
     'Binding of Isaac: Rebirth, Goofball Goals, Who wants to be a Millionaire with rob, Baer',
     'Binding of Isaac: Rebirth, Goofball Goals (Continued), Geoguessr',
     'Binding of Isaac: Rebirth, Goofball Goals, Lethal League',
     'Rebirth, Goofball Goals (continued), Worms: Revolution with rob, Baer',
     'Rebirth, Rebirth, Rebirth.',
     'Binding of Isaac, Goofball Goals, (Continued), Metagame with Rob in Tabletop Simulator',
     'Binding of Isaac, Goofball Goals, (Continued), SpyParty',
     'Binding of Isaac, Vertiginous Golf, (Continued), Spy Party',
     'Binding of Isaac, Goofball Goals',
     'Binding of Isaac, Spelunky (continued), Spy Party',
     'Binding of Isaac, Goofball Goals, (Continued), Worms Revolution with rob and MALF',
     'Binding of Isaac, Bomb Party attempt, Spelunky, Continued, Sonic All Stars Racing with rob and Baer',
     'Binding of Isaac, SpyParty, (Continued), Spelunky',
     'Binding of Isaac, Geoguessr, (Continued), Cards Against Humanity, (continued)',
     'Binding of Isaac, Spyparty, (Continued), Rock of Ages',
     'Binding of Isaac, Choice Chamber, (Continued), Lethal League',
     'Binding of Isaac, Vertiginous Golf, (Continued), Cannon Brawl',
     'Binding of Isaac, Tabletop Simulator - Nicolas Cage Guess who, (Continued), Tabletop Simulator - Metagame, (attempt to play) Family Feud Decades, Spelunky, GeoGeussr',
     'Binding of Isaac, Super Win the Game, Continued, Cards Against Humanity',
     'Binding of Isaac, Rock of Ages, (Continued), Mount your Friends',
     'Master of the Grid, GeoGuessr, (Continued), Binding of Isaac',
     'Binding of Isaac, (Continued), Spelunky, Lethal League, (Continued)',
     'Binding of Isaac, Fenix Rage, (Continued), Gauntlet',
     'Binding of Isaac, Cannon Brawl, (Continued), Lethal League',
     'Binding of Isaac, Spelunky, Continued, Toribash',
     'Binding of Isaac, Mount Your Friends, Continued, Who wants to be a Millionaire?',
     "Binding of Isaac, Cabella's Big Game Hunter: Pro Hunts, Continued, Lethal League",
     'Binding of Isaac, Spelunky, (Continued), SpeedRunners',
     "Binding of Isaac, Rocko's Modern Life, Mount Your Friends, (Continued), Lethal League",
     'Binding of Isaac, Lethal League, Continued, Family Feud Decades',
     'Gang Beasts, Spelunky, (Continued), Crawl',
     'Crawl, Sportsfriends, Cavern Kings, (Continued), Crawl',
     'Binding of Isaac, Mount Your Friends, Spelunky, (Continued), Mount Your Friends',
     'Binding of Isaac, Cavern Kings, (Continued), Screencheat',
     'Binding of Isaac, Cannon Brawl, (Continued), iSketch',
     'Binding of Isaac, Mount Your Friends, (Continued), Toribash',
     'Binding of Isaac, Virtual Bart, Continued, Family Feud Decades',
     'Binding of Isaac, ibb & obb, (Continued), Mount Your Friends',
     'Binding of Isaac, Cannon Brawl, (Continued), Spelunky',
     'Binding of Isaac, Mount Your Friends, (Continued), iSketch',
     'Binding of Isaac, ibb & obb, Worms Revolution',
     'Binding of Isaac, OlliOlli, (Continued), SmartBall',
     'Binding of Isaac, (Continued), Metal Slug 3, XPlane, GeoGuessr, (Continued)',
     'Binding of Isaac, Metal Slug 3, Toribash, GeoGuessr, Toribash',
     'Binding of Isaac, Super House of Dead Ninjas, (Continued), Family Feud Decades',
     'Binding of Isaac, Game Dev Tycoon, Who Wants to be a Millionaire?',
     'Binding of Isaac, Nuclear Throne, (Continued), Family Feud Decades',
     'Binding of Isaac, Bunny Must Die, (Continued), Toribash',
     'News and Fanmail, Spelunky, 100% Orange Juice, (Continued), Cards Against Humanity',
     'News and fanmail, Shovel Knight, (Continued)',
     'News and fanmail, Binding of Isaac, (Continued), Karnov, Magicite',
     'News and fanmail, Binding of Isaac, Magicka Wizard Wars, Family Feud 2010',
     'News and fanmail, Secret Ponchos, (Continued), Cards Against Humanity',
     "News and fanmail, A Wizard's Lizard, (Continued), SpeedRunners, Toribash",
     'News and fanmail, Spelunky, Broforce, (Continued), Binding of Isaac, Magicka Wizard Wars',
     'News and fanmail, Magicka Wizard Wars, (Continued), Family Feud 2010',
     'News and fanmail, Team Fortress 2, (Continued), Worms: Revolution',
     'News and fanmail, Tabletop Simulator, Guess who, (Continued), Trivia, iSketch',
     'News and fanmail, 1001 Spikes, (Continued), Binding of Isaac',
     'News and fanmail, Tabletop Simulator, (Continued), Trivia, Family Feud 2010',
     'News and fanmail, Duck Tails, Continued, London 2012, Spelunky, Toribash',
     'News and fanmail, 100% Orange Juice, (Continued), Toribash',
     'News and fanmail, Toribash, (Continued), Battleblock Theather',
     'News and fanmail, Ascendant, (Continued), Battleblock Theather',
     'News and fanmail, Mendel palace, (Continued), Binding of Isaac, GeoGuessr',
     "News and fanmail, Whomp'em, (Continued), Kero Blaster",
     'News and fanmail, The Addams Family, (Continued), Blade Symphony, Coin Crypt, Cook Serve Delicious',
     'News and fanmail, Family Feud 2010, (Continued), Coin Crypt',
     'News and Fanmail, Dark Souls 2, (Continued), Spelunky, Dark Souls 2',
     'News and fanmail, Dark Souls 2, (Continued), Binding of Isaac',
     'News and fanmail, Dark Souls 2, (Continued), Spelunky',
     'Dark Souls 2, (Continued)',
     'News and fanmail, Family Feud 2010, Worms Revolution',
     'News and fanmail, Aladdin, Continued, Binding of Isaac, Spelunky',
     'Super Castlevania 4, Cook Serve Delicious, Continued, Trivia, Binding of Isaac, Cook Serve Delicious',
     'Chivalry, Family Feud 2010, (Continued), Trivia, Spelunky',
     'BroForce, Family Feud 2010, (Continued), Damned, Spelunky',
     'Family Feud 2010, Cook Serve Delicious, (Continued), Trivia, Spelunky',
     'FTL, Minesweeper, Trivia, Rogue Legacy, Continued, Cook Serve Delicious',
     'Mercenary Kings, Family Feud 2010 (Continued), technical issues, Lightning round and trivia, Spelunky',
     'SpyParty, Revelations 2012, Bomb Party, Trivia, Robot Roller Derby Disco Dodgeball',
     'Choice Chamber, Binding of Isaac, (Continued), Trivia, Spelunky, SpeedRunners',
     'Chivalry, Family Feuds Decades, (Continued), Trivia, Spelunky',
     'Bomb Party, Family Feud 2010, (Continued), Spelunky, Super Meat Boy',
     'Super Mario Bros. The Lost Levels, Bosses Forever, (Continued), Binding of Isaac, Chess',
     'Family Feud 2010, Nuclear Throne, (Continued), Spelunky, Super Meat Boy, Family Feud 2010',
     'Dark Souls, (Continued), Family Feud 2010',
     'Dark Souls, (Continued), Guess Who, Trivia, 10 Second Ninja',
     'Family Feud 2010, Binding of Isaac, (Continued), Trivia, Spelunky',
     'Dark Souls, (Continued), Guess who and trivia, The Yawhg, Spelunky',
     'Dark Souls, (Continued), Guess who and trivia, Binding of Isaac',
     'Dark Souls, (Continued), Guess who and trivia, Family Feud Decades',
     'Megaman 2, Family Feud 2010, (Continued), Trivia, Spelunky',
     'Titanfall, Worms Revolution, (Continued), Guess who and trivia, Spelunky',
     'Titanfall, Family Feud Decades, (Continued), Guess who and trivia, Spelunky',
     'Double Dragon Neon, Family Feud Decades, Bloody Trapland, (Continued), Trivia, Spelunky',
     'Dark Souls, (Continued), (More continued), Trivia, (Continued), Spelunky, (Continued)',
     "Dark Souls, Nagano Winter Olympics '98, (Continued), Guess who and trivia, Family Feud 2010",
     "Who Wants to be a Millionaire?, Lillehammer '94 Winter Olympics (continued), Guess who and trivia, Spelunky",
     'Magicka Wizard Wars, Family Feud 2010',
     'Loadout, Family Feud 2014, (Continued)',
     'Family Feud 2010, Octodad: Dadliest Catch, (Continued), Guess who and trivia, Spelunky',
     "Kirby's Dream Course, Family Feud 2010",
     'Family Feud 2010, Guess who and trivia, Spelunky (Return the slab run)',
     'Donkey Kong Country race, Worms Revolution (Continued), Guess who, trivia, Spelunky',
     'Are You Smarter Than a Fifth Grader?, Family Feud 2010',
     'Who Wants to be a Millionaire?, Chivalry',
     'Nidhogg, Family Feud 2010, (Continued)',
     'Bloody Trapland, Long Live the Queen, Family Feud 2010, (Continued), Guess who, trivia, Robot Roller Derby Disco Dodgeball',
     'Family Feud 2010, Cook Serve Delicious, (Continued), Guess who and trivia, Binding of Isaac',
     'Family Feud 2010, Gunman Clive Race, (Continued), Guess who and trivia, Spelunky',
     'Just Cause 2, SpeedRunners, SpeedRunners continued, Guess who, trivia, Family Feud 2010',
     'The Politicial Machine 2012, Talisman (continued), Guess who and trivia, Spelunky',
     'Family Feud 2010, Super Crate Box (continued), Guess who and trivia, Spelunky',
     'Starbound, Family Feud 2010 (continued), Guess who, trivia, Spelunky',
     'Starbound, Tiny Brains (continued), Guess who and trivia, Spelunky',
     'Starbound, Binding of Isaac (continued), Guess who and trivia, Spelunky',
     'Frozen Endzone, Starbound (continued), Guess who and trivia, Spelunky',
     'Starbound, Family Feud 2010 (continued), Guess who and trivia, Spelunky',
     'Incredipede, Starbound (continued), Guess who and trivia, Spelunky',
     'Cook Serve Delicious, SpyParty (continued), Guess who and trivia, Spelunky',
     'Painter, Divekick, Guess who and trivia, Spelunky',
     'Dark Souls Invasions, Family Feud 2010, Guess who and trivia (continued), Spelunky',
     'VVVVVV, SpeedRunners (Continued), Guess who and trivia, Spelunky',
     'Family Feud 2010, Chivalry (continued), Guess who and trivia, Spelunky',
     'Robot Roller Derby Disco Dodgeball, Family Feud 2010 (continued), Guess who and trivia, Spelunky',
     'Chivalry, Salty bet (continued), Guess who and trivia, Spelunky',
     'iSketch, Family Feud 2010 (continued), Guess who and trivia, Spelunky',
     'Counterstrike: Global Offensive, Payday 2 (continued), Guess who and trivia, Spelunky',
     'Final exam, Sonic All Stars Racing Transformed (continued), Guess who and trivia, Spelunky',
     'Call of Duty: Ghosts, NLSS Videogame Olympics HD: SpeedRunners (continued), Guess who, trivia, Spelunky',
     'Who Wants to be a Millionaire?, Worms Revolution (continued), Guess who and trivia, Spelunky',
     'Left 4 Dead, NLSS Videogame Olympics HD: Hotline Miami, Trivia and Guess Who, Speedrunners',
     'Chivalry, NLSS Videogame Olympics HD: The Typing of the Dead: Overkill (continued), Guess who and Trivia, Spelunky',
     'Who Wants to be a Millionaire?, NLSS Videogame Olympics HD: Probably Archery, Guess who and trivia, Spelunky, SpeedRunners',
     'Chivalry, NLSS Videogame Olympics HD: Dark Souls race (continued), Guess who and trivia, Spelunky, SpeedRunners',
     'Who Wants to be a Millionaire?, NLSS Videogame Olympics HD: Mount your friends, Guess who and trivia, Spelunky, SpeedRunners',
     'Chivalry: Deadliest Warrior, NLSS Videogame Olympics HD: Boardgame online, Guess Who and Trivia, Spelunky',
     'Who Wants to be a Millionaire?, NLSS Videogame Olympics HD: Worms Revolution, Guess who and trivia, Spelunky, SpeedRunners',
     'Who Wants to be a Millionaire?, NLSS Videogame Olympics HD: Pokemon Snap, Guess who and trivia, Spelunky',
     'Terraria, NLSS Videogame Olympics HD: Papers Please, Guess who and trivia, Spelunky, SpeedRunners',
     'Surgeon Simulator 2013, SpeedRunners, Guess who and Trivia, Spelunky',
     'Cook Serve Delicious, Spelunky, Metagame and Trivia, More Spelunky, SpeedRunners',
     'Spelunky, Super Meat Boy race, Metagame and Trivia, More Spelunky, Speedrunners',
     'SpeedRunners, Monaco, Metagame and trivia, Spelunky',
     'Spelunky, War of the Vikings, SpyParty, More War of the Vikings, Metagame and trivia, Spelunky',
     'Spelunky, SpeedRunners, Metagame and Trivia (continued), Short Spelunky, More SpeedRunners',
     'SpeedRunners, The Showdown Effect, Hearthstone, Metagame and Trivia, Spelunky',
     'Spelunky, Worms Revolution, Speedrunners, Speedrunners continued, Metagame and trivia, (Short) Spelunky, Speedrunners',
     'Spelunky, Hearthstone, Speedrunners, The Metagame + Trivia, Spelunky, More Speedrunners',
     'Trine 2, Game Dev Tycoon, Trivia, Spelunky',
     'Spelunky, Sonic All-Star Racing, Trivia, More Spelunky',
     'Spelunky, Sonic All-Stars Racing, Trivia, More Spelunky',
     'Dark Souls invasions, Binding of Isaac, Trivia, Spelunky',
     'Spelunky, Volgarr the Viking, Trivia, More Spelunky',
     'Spelunky, Binding of Isaac, Trivia, More Spelunky',
     'Dark Souls invasions, Family Feud 2010, Trivia, Spelunky',
     'Spelunky, Trivia, Splinter Cell: Blacklist, More Spelunky',
     'Spelunky, War of the Vikings, Trivia, More Spelunky',
     'Spelunky, Splinter Cell: Black Ops, Trivia, More Spelunky',
     'Spelunky, Divekick, Trivia, Splinter Cell: Blacklist',
     'Dark Souls invasions, Trivia, Spelunky',
     'Dark Souls invasions, Trivia, Risk of Rain, Spelunky',
     'Dark Souls invasions, Trivia, Spelunky',
     'Dark Souls invasions, Trivia, Spelunky',
     'Spelunky, Trivia, Worms Revolution',
     'Spelunky, Trivia, Family Feud 2010',
     'Spelunky, Trivia, Payday 2',
     'Dark Souls invasions, Trivia, The Price is Right',
     'Dark Souls invasions, Spelunky, Trivia, Family Feud 2010',
     'Dark Souls invasion, Trivia, Family Feud 2010',
     'Dark Souls invasions, Trivia, Family Feud 2010',
     'Dark Souls invasions, Trivia, Family Feud 2010',
     'Dark Souls invasions, Trivia, Super Mutant Ninja Turtles IV: Turtles in Time race, Super Meat Boy race',
     'Dark Souls invasions, Trivia, Megaman X race',
     'Dark Souls invasions, Trivia, Super Mario Bros. 3 race',
     'Dark Souls invasions, Trivia, Runner 2 race',
     'Dark Souls Invasions, Trivia, iSketch',
     'Dark Souls invasions, Trivia, Super Meat Boy race',
     'Dark Souls invasions, Trivia, iSketch',
     'Dark Souls invasions, Skulls of the Shogun, Trivia, iSketch',
     'Dark Souls invasions, Scrolls, Trivia, (Aborted attempt to play) Arma III, CS:GO',
     'Dark Souls invasions, Trivia, iSketch',
     'Dark Souls invasions, Mount Your Friends, Trivia, Boardgame online',
     'Dark Souls invasions, Trivia, DOTA 2',
     'Dark Souls, SpyParty, Trivia, War Thunder',
     'Dark Souls, Gunpoint, Trivia, Cards Against Humanity',
     'Dark Souls, Trivia, iSketch',
     'Dark Souls Invasions, Trivia, Crusader Kings II - The Old Gods',
     'Dark Souls invasions, Trivia, Dota 2',
     'Dark Souls invasions, SpyParty, Trivia, Dota 2',
     'Dark Souls Invasions, Towerfall, Trivia, iSketch',
     'Dark Souls invastions, Paranautical Activity, Trivia, DOTA 2',
     'Dark Souls soul level 1 play, Trivia with Ohmwrecker, iSketch',
     'Dark Souls soul level 1 play, Trivia, Sanctum 2, Counterstrike',
     'Dark Souls invasions, Project Zomboid, Trivia, iSketch',
     "Dark Souls soul level 1 play, Don't Starve, Trivia, Counterstrike: Global Offensive",
     'Dark Souls invasions, Rogue Legacy, Trivia, Leviathan Warships',
     'Dark Souls invasions, Game Dev Tycoon, Trivia, iSketch',
     'Ask me anything, Dark Souls invasions, Trivia, Cards Against Humanity',
     'Dark Souls invasions, Trivia, Battle block theater',
     'Ask me anything, Dark Souls Invasions, Trivia, Boardgame online, iSketch',
     'Dark Souls invasions, La Mulana, Trivia, Cards Against Humanity',
     'Dark Souls invasions, Trivia, Boardgame online',
     'Dark Souls Invasions, Risk of Rain with Paul Morse and Dunkin Drummand, Trivia, iSketch',
     'Dark Souls Invasions, Trivia, Cards Against Humanity',
     'Dark Souls DLC, invasions, Trivia, Ask me anything, DOTA 2, iSketch',
     'Dark Souls DLC, Trivia, Strange Loves Vampire Boyfriends, Trackmania 2',
     'Dark Souls DLC, Trivia, Cards Against Humanity',
     'Dark Souls DLC, Trivia, Cards Against Humanity',
     'Dark Souls Invasions, Trivia, iSketch, Ask me anything on Twitter',
     'Dark Souls Invasions, Trivia, Cards Against Humanity',
     'Dark Souls Invasions, Trivia, Arma III, Ask me anything on Twitter',
     'Dark Souls Invasions, Trivia, Rollercoaster Tycoon, More Dark Souls Invasions, Ask me anything on Twitter',
     'Dark Souls Invasions, Trivia, Worms Revolution, Ask me anything on Twitter',
     'Dark Souls Invasions, Trivia, The Showdown Effect',
     'Dark Souls Invasions, Trivia, More Dark Souls Invasions, Arma III, Ask me anything on Twitter',
     'Dark Souls invasions, Trivia, Tomb Raider, Ask me anything on Twitter',
     "Delver's Drop with Ryan Baker and Ryan Burrell, Dark Souls, Trivia, More Dark Souls, Ask me anything on Twitter",
     'Dark Souls, Trivia, Trials Evolution, Ask me anything on Twitter',
     'Dark Souls, Trivia, More Dark Souls, Ask me anything on Twitter',
     'Runner 2, Super House of Dead Ninjas, Trivia, Hitman: Bloodmoney, Ask me anything on Twitter']



I'll have to clean these up later to make sure all the games are spelled consistantly.


```python
#Number of dockets, not individual games
len(games)
```




    588



Now lets take a look at the first lines of the text file which contain the date of the show and the people who joined the show that day. They are seperated in the file by ().


```python
date_crew = []
for s in shows:
    dc = s.split('\n')[0]
    date_crew.append(dc)
print(date_crew)
```

    ['(August 24, 2017) (NL, RLS, CS, rob)', '(August 23, 2017) (NL, RLS, rob w/ Baer, LGW, HCJ)', '(August 21, 2017) (NL, RLS, JS, rob)', '(August 17, 2017) (NL w/ Sin, RLS, LGW, HCJ, Baer)', '(August 16, 2017) (NL, RLS w/ rob, Baer, LGW, Dan)', '(August 14, 2017) (NL, JS, MALF, LGW w/ Baer, HCJ)', '(August 10, 2017) (NL, RLS, rob, LGW)', '(August 7, 2017) (NL, RLS, JS, rob w/ Baer)', '(August 3, 2017) (NL, RLS, CS w/ rob, MALF)', '(August 2, 2017) (NL, RLS, LGW w/ Baer, Kory)', '(July 31, 2017) (NL, RLS, JS, rob w/ Sin, Baer, TB)', '(July 27, 2017) (NL, RLS, CS w/ MALF)', '(July 26, 2017) (NL, RLS w/ LGW, Sin, Baer, Dan)', '(July 24, 2017) (NL, RLS, JS w/ LGW)', '(July 20, 2017) (NL, RLS, CS, LGW w/ Baer)', '(July 19, 2017) (NL, RLS, LGW w/ MALF, rob, Baer)', '(July 13, 2017) (NL, RLS, rob w/ LGW, Baer)', '(July 12, 2017) (NL, RLS, rob w/ Baer)', '(July 10, 2017) part 1, part 2 (NL, RLS, JS w/ rob, Sin)', '(July 6, 2017) (NL, RLS, CS, rob w/ Baer)', '(July 5, 2017) (NL, RLS, rob w/ LGW, Baer)', '(July 3, 2017) (NL, RLS, rob w/ Sin, LGW, Baer)', '(June 22, 2017) (NL, RLS, CS, rob w/ JS)', '(June 21, 2017) Nick view (NL, RLS, rob w/ LGW, Baer)', '(June 19, 2017) Nick view (NL, RLS, JS w/ rob, Kate, Baer)', 'Solo (June 15, 2017) (NL w/ MALF, rob)', 'Solo (June 14, 2017) (NL w/ MALF, LGW, JS, Baer)', 'NLSS Masters (June 12, 2017) (NL, RLS, JS, MALF)', '(June 8, 2017) (NL, RLS w/ rob, MALF, Baer)', '(June 7, 2017) (NL, RLS, rob w/ Dan, Baer, Sin, Blueman)', '(June 5, 2017) Nick view (NL, RLS, JS w/ MALF)', '(June 1, 2017) Nick view (NL, RLS, CS w/ rob, Baer)', '(May 31, 2017) (NL, rob, LGW, Baer)', '(May 29, 2017) (NL, RLS, JS w/ rob, MALF)', '(May 25, 2017) (NL, RLS, rob w/ MALF, Baer)', '(May 24, 2017) (NL, RLS, rob, LGW w/ MALF, Dan)', '(May 22, 2017) (NL, RLS, JS w/ MALF)', '(May 18, 2017) (NL, RLS, CS, rob w/ Kate, Baer)', '(May 17, 2017) (NL, RLS w/ Dan, LGW)', '(May 15, 2017) (NL, RLS, JS w/ rob, LGW, MALF)', '(May 11, 2017) (NL, RLS, rob, Baer w/ LGW, Sin)', '(May 10, 2017) (NL, RLS, rob w/ Baer, LGW, Dan)', '(May 8, 2017) (NL, RLS w/ rob, MALF)', '(May 4, 2017) (NL, RLS, rob w/ MALF, Baer)', '(May 3, 2017) (NL, RLS, rob w/ MALF)', '(May 1, 2017) (NL, RLS, JS w/ rob, MALF)', '(April 27, 2017) (NL, RLS w/ JS, MALF, Baer, LGW, Kate)', '(April 26, 2017) (NL, RLS, rob, LGW w/ Baer)', '(April 24, 2017) (NL, RLS, rob, LGW w/ Baer)', '(April 20, 2017) Nick view (NL, RLS, JS, CS w/ rob)', '(April 19, 2017) Nick view (NL, RLS, LGW w/ rob, Baer)', '(April 6, 2017) part 1 part 2 Nick view (NL, RLS, CS w/ LGW)', '(April 5, 2017) (NL, RLS, rob w/ Baer)', '(April 3, 2017) (NL, JS w/ MALF, LGW, rob, Baer, Sin)', '(March 30, 2017) (NL, RLS, CS w/ BaerBaer, MALF)', '(March 29, 2017) Nick Video (NL, RLS, rob, LGW w/ Dan)', '(March 27, 2017) (NL, RLS, JS w/ rob)', '(March 23, 2017) (NL, RLS, CS w/ MALF)', '(March 22, 2017) Nick Video (NL, RLS, LGW w/ MALF, Baer)', '(March 20, 2017) (NL, RLS, LGW w/ Baer, rob)', '(March 16, 2017) (NL, CS, LGW w/ MALF, Baer)', '(March 15, 2017) (NL, RLS, LGW w/ Kate)', '(March 8, 2017) (NL, RLS, rob w/ LGW, Baer, Sin)', '(March 6, 2017) (NL, RLS, JS w/ LGW, Baer, Sin)', '(March 1, 2017) (NL, rob, LGW w/ MALF, Baer)', '(February 27, 2017) (NL, RLS, JS w/ rob, LGW, Baer)', '(February 23, 2017) (NL, RLS, MALF w/ LGW, Baer)', '(February 22, 2017) (NL, RLS, rob, LGW w/ Dan)', '(February 20, 2017) Nick view (NL, RLS, JS, LGW w/ rob, Baer, Sin)', '(February 16, 2017) (NL, RLS, LGW w/ rob, Baer, MALF)', '(February 15, 2017) (NL, RLS, rob, LGW w/ Mathas)', '(February 13, 2017) (NL, JS w/ rob, LGW)', '(February 9, 2017) (NL, RLS, rob, LGW w/ MALF, Baer)', '(February 8, 2017) part 1 part 2 (NL, RLS, CS w/ rob, LGW, MALF, Baer)', '(February 6, 2017) (NL, RLS, JS w/ MALF)', '(February 2, 2017) (NL, RLS, rob, LGW w/ Baer, MALF)', '(February 1, 2017) (NL, RLS, rob, LGW)', '(January 31, 2017) (NL, JS, MALF, rob w/ Sin)', '(January 26, 2017) (NL, MALF, rob, LGW)', '(January 25, 2017) (NL, rob, LGW w/ MALF, Dan, Sin, Kate)', '(January 23, 2017) part 1 part 2 (NL, JS, MALF)', '(January 19, 2017) (NL, MALF, rob, LGW)', '(January 18, 2017) (NL, rob, LGW w/ MALF)', '(January 16, 2017) part 1 part 2 (NL, JS, MALF w/ LGW, rob, Baer, Sin, Dan)', '(January 12, 2017) (NL, RLS, MALF w/ LGW, Crendor)', '(January 11, 2017) (NL, RLS w/ rob, LGW, Dan)', '(January 9, 2017) (NL, RLS, JS w/ Baer, rob, LGW)', '(January 5, 2017) (NL, RLS, rob w/ MALF, Baer, rob)', '(January 4, 2017) part 1 part 2 (NL, RLS, rob w/ MALF, LGW)', '(January 2, 2017) (NL, RLS, JS w/ LGW)', '(December 29, 2016) (NL, RLS, LGW w/ rob, Baer, JS)', '(December 28, 2016) (NL, RLS, LGW w/ rob, Baer)', '(December 26, 2016) (NL, RLS, JS w/ rob, Sin)', 'Bootleg (December 22, 2016) part 1 part 2 part 3 (NL, LGW w/ Kate)', '(December 21, 2016) (NL, RLS, CS w/ Baer, rob, LGW, Dan)', '(December 19, 2016) (NL, RLS, JS, rob w/ LGW)', '(December 15, 2016) (NL, RLS, MALF, rob w/ Kate, LGW)', '(December 14, 2016) (NL, RLS, CS w/ LGW)', '(December 12, 2016) (NL, RLS, JS, LGW)', '(December 8, 2016) (NL, RLS, MALF, rob w/ LGW)', '(December 7, 2016) (NL, RLS, rob, LGW w/ MALF)', '(December 6, 2016) (NL, RLS, MALF, LGW w/ rob)', '(December 5, 2016) (NL, RLS, JS, rob w/ LGW, Baer)', '(November 30, 2016) (NL, RLS, rob, LGW)', '(November 28, 2016) (NL, RLS, rob, LGW w/ Baer)', '(November 24, 2016) (NL, MALF, rob, LGW)', '(November 23, 2016) (NL, RLS, CS w/ rob, LGW, Dan, MALF, Baer)', '(November 21, 2016) (NL, RLS, rob, LGW w/ Baer, Sin)', '(November 17, 2016) (NL, RLS, MALF, rob w/ Baer, LGW)', '(November 16, 2016) (NL, RLS, CS w/ Mathas, rob)', '(November 14, 2016) (NL, RLS, JS, rob w/ LGW, BRex)', '(November 10, 2016) (NL, RLS, rob, LGW w/ JS, MALF)', '(November 9, 2016) (NL, JS, MALF, rob, LGW)', '(November 7, 2016) (NL, RLS, JS, LGW)', '(November 3, 2016) (NL, RLS w/ rob, LGW)', '(November 2, 2016) Nick view (NL, RLS w/ MALF, rob, LGW, Dan)', '(October 31, 2016) (NL, RLS, JS, Sin w/ rob, LGW)', '(October 27, 2016) (NL, RLS, MALF w/ LGW, rob)', '(October 26, 2016) Nick view (NL, RLS, CS, rob w/ LGW)', '(October 24, 2016) (NL, RLS, JS, LGW w/ rob, Baer)', '(October 20, 2016) (NL, RLS, MALF w/ rob, Baer, LGW)', '(October 19, 2016)(NL, RLS, CS, rob w/ Baer)', '(October 17, 2016) (NL, RLS, JS, MALF w/ rob, LGW, Baer)', '(October 13, 2016) (NL, RLS w/ rob, LGW)', '(October 12, 2016) (NL, RLS, CS, LGW w/ MALF, rob, Baer)', '(October 10, 2016) (NL, RLS, JS, rob w/ LGW, GhostBill)', '(October 6, 2016) (NL, RLS, MALF, LGW w/ rob, Baer)', '(October 5, 2016) (NL, RLS, rob, LGW w/ Baer, Sin)', '(October 3, 2016) (NL, RLS, JS w/ rob)', '(September 29, 2016) (NL, RLS, rob, LGW)', '(September 28, 2016) (NL, RLS, CS w/ rob, LGW)', '(September 26, 2016) (NL, RLS, JS w/ rob, Kate)', '(September 22, 2016) (NL, RLS, MALF w/ rob, LGW)', '(September 12, 2016) (NL, RLS, JS w/ MALF, rob, LGW)', '(September 10, 2016) (NL, RLS, MALF w/ rob, Baer)', '(September 8, 2016) (NL, RLS, MALF w/ alpacapatrol, LGW, Sin)', 'Solo (September 7, 2016) (NL w/ Sin, MALF, LGW, Baer, Dan, rob)', '(August 31, 2016) (NL, RLS w/ rob, LGW, Dan)', '(August 29, 2016) Nick view (NL, RLS, JS w/ rob, LGW)', '(August 25, 2016) (NL, RLS w/ rob, MALF, LGW)', '(August 24, 2016) (NL, MALF w/ rob, LGW, dan)', '(August 22, 2016) (NL, MALF w/ rob, LGW)', '(August 18, 2016) (NL, RLS w/ JS, MALF, rob, LGW)', '(August 17, 2016) (NL, RLS, CS w/ rob, Baer, LGW, Dan)', '(August 15, 2016) part 1 part 2 Nick view (NL, RLS, JS w/ rob, Baer, LGW)', '(August 10, 2016) (NL, RLS w/ rob, LGW)', '(August 8, 2016) (NL, RLS, JS w/ LGW, Baer)', '(August 4, 2016) Nick view (NL, RLS, MALF w/ rob, LGW, Sin, Baer)', '(August 3, 2016) part 1 part 2 (NL, RLS, CS w/ rob, LGW)', '(August 1, 2016) (NL, RLS, JS w/ Baer, LGW)', 'Bootleg (July 28, 2016) (NL, MALF w/ JS, LGW)', 'Solo (July 27, 2016) (NL, MALF)', 'Solo (July 25, 2016) (NL)', '(July 21, 2016) (NL, RLS w/ LGW, Sin, Mathas)', '(July 20, 2016) (NL, RLS, CS w/ Baer, LGW)', '(July 18, 2016) Nick view (NL, RLS w/ MALF, LGW)', '(July 14, 2016) (NL, RLS w/ LGW, Sin)', '(July 13, 2016) (NL, RLS, CS w/ Sin)', '(July 11, 2016) (NL, RLS w/ Baer, rob, LGW)', 'Solo (July 7, 2016) (NL w/ rob)', '(July 6, 2016) (NL, RLS, CS w/ LGW)', '(July 4, 2016) (NL, RLS, JS w/ rob, LGW)', '(June 30, 2016) (NL, RLS w/ MALF, rob, Baer, LGW)', '(June 29, 2016) (NL, RLS, CS w/ rob, LGW)', '(June 27, 2016) (NL, RLS, JS w/ rob, LGW)', '(June 23, 2016) (NL, RLS w/ rob, MALF, Dan)', '(June 20, 2016) (NL, RLS, JS w/ rob, Mathas)', '(June 9, 2016) Nick view (NL, RLS, rob w/ LGW)', '(June 8, 2016) (NL, RLS, CS w/ rob, LGW)', '(June 6, 2016) (NL, RLS, JS w/ LGW, MALF, Dan)', '(June 2, 2016) (NL, RLS w/ rob, LGW)', '(June 1, 2016) (NL, RLS, CS w/ Baer, LGW)', '(May 30, 2016) (NL, RLS, JS w/ rob, Baer, LGW)', '(May 26, 2016) (NL, RLS w/ rob, LGW, Baer)', '(May 25, 2016) (NL, RLS, CS w/ rob, MALF)', '(May 23, 2016) (NL, JS w/ rob, Dan, Sin, LGW)', 'Bootleg Solo (May 19, 2016) (NL w/ Mathas, rob, LGW, Sin)', 'Bootleg Solo (May 18, 2016) (NL w/ Sin, rob, LGW)', 'Bootleg (May 16, 2016) (NL, RLS, JS w/ rob, LGW)', '(May 12, 2016) Nick view (NL, RLS w/ rob, LGW)', '(May 11, 2016) (NL, RLS, CS w/ rob)', 'Solo (May 9, 2016) (NL w/ Sin, rob, LGW)', '(May 5, 2016) (NL, RLS w/ JS, rob)', '(May 4, 2016) (NL, RLS, rob w/ MALF)', '(May 2, 2016)(NL, RLS, JS w/ MALF)', 'Solo (April 21, 2016) (NL w/ Sin, rob, LGW)', 'Solo (April 20, 2016) (NL w/ Sin)', '(April 18, 2016) (NL, RLS, JS)', 'Bootleg (April 14, 2016) (NL, RLS w/ MALF, rob, LGW)', '(April 13, 2016) (NL, RLS w/ rob, MALF, Dan, LGW)', '(April 11, 2016) (NL, RLS, JS)', '(April 7, 2016) (NL, RLS w/ MALF, rob, LGW)', '(April 6, 2016) (NL, RLS w/ LGW)', '(April 4, 2016) (NL, RLS, JS w/ rob)', '(March 31, 2016) (NL, RLS w/ MALF, rob, LGW)', '(March 30, 2016) (NL, RLS, CS w/ JS, rob)', '(March 28, 2016) (NL, RLS w/ MALF, rob)', '(March 24, 2016) part 1 part 2 Nick view (NL, RLS w/ LGW)', '(March 23, 2016) part 1 part 2 (NL, RLS, CS w/ MALF, rob)', 'Bootleg Solo (March 3, 2016) (NL, RLS w/ rob, Dan, LGW)', '(March 2, 2016) (NL, RLS, CS w/ MALF)', '(February 29, 2016) (NL, RLS, JS w/ rob)', '[3 year NLversary!] (February 25, 2016) (NL, RLS, dan)', '(February 24, 2016) (NL, RLS, CS)', '(February 22, 2016) (NL, RLS, JS w/ rob)', '(February 18, 2016) (NL, RLS)', 'Bootleg Solo (February 17, 2016) (NL)', '(February 15, 2016) (NL, RLS, JS w/ MALF)', '(February 10, 2016) (NL, RLS, CS w/ MALF)', '(February 8, 2016) (NL, RLS, JS)', '(February 4, 2016) (NL, RLS w/ MALF)', '(February 3, 2016) (NL, RLS, CS w/ MALF)', '(February 1, 2016) (NL, RLS, JS)', '(January 28, 2016) (NL, RLS, MALF)', '(January 27, 2016) (NL, RLS)', '(January 25, 2016) (NL, RLS, JS, MALF)', '(January 21, 2016) (NL, RLS, MALF w/ rob)', '(January 20, 2016) (NL, RLS w/ rob)', '(January 18, 2016) (NL, RLS, JS)', '(January 13, 2016) (NL, RLS, CS w/ rob)', '(January 11, 2016) (NL, RLS, JS w/ rob)', '(January 7, 2016) (NL, RLS w/ MALF, rob)', '(January 6, 2016) (NL, RLS, CS w/ rob)', '(January 4, 2016) Nick view (NL, RLS, JS w/ rob)', '(December 31, 2015) (NL, RLS w/ rob)', '(December 30, 2015) (NL, RLS, CS)', '(December 28, 2015)(NL, RLS, JS)', 'Solo (December 24, 2015) (NL w/ Kate)', '(December 23, 2015) part 1 part 2 (NL, RLS, CS)', '(December 21, 2015) (NL, RLS, JS, w/ MALF)', 'Solo (December 19, 2015) (NL w/ Kate, Baer)', 'Solo (December 18, 2015) (NL w/ Kate)', '(December 10, 2015) (NL, RLS, MALF)', '(December 9, 2015) (NL, RLS, CS w/ rob)', '(December 7, 2015) part 1 part 2 (NL, RLS)', '(December 2, 2015) (NL, RLS, CS w/ rob)', '(November 30, 2015) part 1 part 2 (NL, RLS, MALF w/ rob)', 'Solo (November 26, 2015) (NL)', '(November 25, 2015) (NL, RLS, CS w/ rob)', '(November 23, 2015) (NL, RLS, JS w/ rob)', '(November 19, 2015) (NL, RLS, MALF)', '(November 18, 2015) part 1 part 2 (NL, RLS w/ Baer, rob)', '(November 16, 2015) (NL, RLS, JS)', '(November 12, 2015) (NL, RLS w/ rob, Baer)', '(November 11, 2015) (NL, RLS, CS w/ rob)', '(November 9, 2015) (NL, RLS, JS w/ Baer)', '(November 5, 2015) (NL, RLS, MALF)', '(November 4, 2015) (NL, RLS, CS)', '(November 2, 2015) part 1 part 2 (NL, RLS, JS)', '(October 29, 2015) (NL, RLS)', 'Bootleg Solo (October 28, 2015) part 1, part 2, part 3 (NL)', 'Bootleg (October 26, 2015) part 1 part 2 part 3 (NL, JS, MALF)', 'Bootleg Solo (October 22, 2015) part 1, part 2, part 3 (NL)', 'Solo (October 21, 2015) part 1 part 2 (NL)', '(October 19, 2015) part 1 part 2 (NL, JS, MALF)', '(October 15, 2015) part 1 part 2 (NL, MALF)', '(October 14, 2015) part 1 part 2 (NL, MALF)', 'October 7, 2015 (NL, RLS, CS w/ rob)', '(October 5, 2015) part 1 part 2 (NL, RLS, JS w/ rob)', '(October 1, 2015) part 1 part 2 (NL, RLS w/ rob, Dan)', '(September 30, 2015) part 1 part 2 (NL, RLS, CS w/ rob)', '(September 28, 2015) part 1 part 2 (NL, RLS, JS)', '(September 24, 2015) (NL, RLS)', '(September 23, 2015) (NL, RLS, CS w/ rob)', '(September 21, 2015) (NL, RLS, JS w/ rob)', '(September 17, 2015) part 1 part 2 Nick view (NL, RLS)', '(September 16, 2015) part 1 part 2 (NL, RLS, CS)', '(September 14, 2015) part 1 part 2 (NL, RLS, JS)', '(September 10, 2015) (NL, RLS)', '(September 9, 2015) part 1 part 2 (NL, RLS, CS w/ rob)', '(September 7, 2015) part 1 part 2 (NL, RLS, JS)', '(September 3, 2015) part 1 part 2 (NL, RLS w/ Baer, MALF)', '(September 2, 2015) part 1 part 2 Nick view (NL, RLS, CS w/ rob)', '(August 24, 2015) (NL, RLS, JS)', '(August 20, 2015) part 1 part 2 (NL, RLS)', '(August 19, 2015) part 1 part 2 Nick view (NL, RLS, CS w/ rob)', '(August 17, 2015) part 1 part 2 Nick view (NL, RLS, JS)', '(August 13, 2015) part 1 part 2 Nick view (NL, RLS, rob)', '(August 12, 2015) part 1 part 2 Nick view (NL, RLS, CS w/ JS)', '(August 10, 2015) part 1 part 2 part 3 Nick view (NL, RLS, JS)', 'Bootleg (August 6, 2015) part 1, part 2 (NL, RLS w/ rob, Baer, MALF)', '(August 5, 2015) part 1 part 2 Nick view (NL, RLS, CS)', '(August 3, 2015) part 1 part 2 (NL, RLS, JS w/ rob)', '(July 30, 2015) part 1 part 2 Nick view (NL, RLS)', '(July 29, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Mathas)', '(July 27, 2015) part 1 part 2 (NL, RLS, JS)', '(July 16, 2015) part 1 part 2 Nick view (NL, RLS w/ Brex)', '(July 15, 2015) part 1 part 2 Nick view (NL, RLS, CS)', '(July 13, 2015) part 1 part 2 Nick view (NL, RLS, JS w/ Baer)', '(July 9, 2015) part 1 part 2 part 3 Nick view (NL, RLS w/ Baer, MALF)', '(July 8, 2015) Part 1 Part 2 (NL, RLS, CS)', '(July 6, 2015) part 1 part 2 (NL, RLS, JS)', '(July 2, 2015) part 1 part 2 Nick view (NL, RLS w/ rob)', '(July 1, 2015) Nick view (NL, RLS, CS w/ rob)', '(June 29, 2015) part 1 part 2 Nick view (NL, RLS, JS)', '(June 25, 2015) Nick view (NL, RLS w/ rob)', '(June 24, 2015) Nick view (NL, RLS, CS w/ rob)', '(June 22, 2015) part 1 part 2 Nick view (NL, RLS, JS)', '(June 18, 2015) part 1 part 2 (NL, RLS)', '(June 17, 2015) part 1 part 2 (NL, RLS)', '(June 15, 2015) part 1 part 2 (NL, RLS, JS w/ MALF)', '(June 11, 2015) part 1 part 2 (NL, RLS w/ rob, MALF)', '(June 10, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(June 8, 2015) part 1 part 2 (NL, RLS w/ rob)', '(May 28, 2015) part 1 part 2 (NL w/ Baer, Fox)', '(May 27, 2015) part 1 part 2 (NL)', '(May 25, 2015) part 1 part 2 (NL, Arumba)', '(May 21, 2015) part 1 part 2 (NL, RLS)', '(May 20, 2015) part 1 part 2 (NL, RLS)', 'Bootleg (May 18, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', 'Bootleg (May 14, 2015) part 1 part 2 part 3 (NL, RLS)', 'Bootleg (May 13, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', 'Bootleg (May 11, 2015) part 1 part 2 part 3 (NL, RLS)', '(May 7, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(May 6, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(May 4, 2015) Part 1 part 2 (NL, RLS w/ rob, Baer)', 'Bootleg Solo (April 23, 2015) part 2 part 3 (NL)', 'Bootleg (April 22, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', '(April 20, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', 'Bootleg (April 16, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', '(April 15, 2015) part 1 part 2 (NL, RLS w/ cobaltstreak, baer, rob)', 'Bootleg (April 13, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', '(April 9, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(April 8, 2015) part 1 part 2 part 3 (NL, RLS w/ rob)', '(April 6, 2015) part 1 part 2 (NL, RLS)', '(April 2, 2015) part 1 part 2 (NL, RLS w/ Baer)', '(April 1, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(March 26, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', 'Bootleg (March 25, 2015) part 1 part 2 part 3 (NL, RLS)', '(March 23, 2015) part 1 part 2 (NL, RLS)', '(March 19, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(March 18, 2015) part 1 part 2 (NL, RLS)', 'Solo (March 12, 2015) part 1 part 2 (NL)', '(March 11, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', 'Bootleg (February 26, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, fox)', '[2 year NLversary!] Bootleg (February 25, 2015) part 1 part 2 part 3 (NL, RLS w/ JS, rob)', '(February 23, 2015) part 1 part 2 (NL, RLS w/ JS, rob)', 'Bootleg (February 19, 2015) part 1 part 2 part 3 (NL, RLS w/ rob)', '(February 18, 2015) part 1 part 2 (NL, RLS w/ JS, rob)', '(February 16, 2015) part 1 part 2 (NL, RLS w/ JS, rob)', '(February 12, 2015) part 1 part 2 (NL, RLS)', '(February 11, 2015) part 1 part 2 (NL w/ Baer)', '(February 9, 2015) part 1 part 2 (NL, RLS)', '(February 5, 2015) part 1 part 2 part 3 (NL, RLS w/ JS, rob)', '(February 2, 2015) part 1 part 2 (NL, RLS w/ rob)', '(January 29, 2015) part 1 part 2 (NL, RLS w/ rob)', '(January 28, 2015) part 1 part 2 (NL, RLS)', 'Bootleg (January 8, 2015) part 1, part 2, part 3 (NL, RLS)', '(January 7, 2015) part 1 part 2 (NL, RLS w/ JS, rob)', '(January 5, 2015) part 1 part 2 (NL, RLS w/ rob)', '(January 1, 2015) part 1 part 2 (NL, RLS)', '(December 31, 2014) part 1 part 2 (NL, RLS)', '(December 29, 2014) part 1, part 2 (NL, RLS w/ fox)', 'Bootleg (December 22, 2014) part 1, part 2 (NL, RLS)', '(December 18, 2014) part 1, part 2 (NL)', '(December 15, 2014) part 1, part 2 (NL, RLS)', 'Bootleg (December 11, 2014) part 1 part 2 (NL, RLS w/ Kate, Baer)', '(December 10, 2014) part 1, part 2 (NL, RLS)', '(December 8, 2014) part 1, part 2 (cat cam!) (NL, RLS w/ rob, Mag)', '(December 4,2014) part 1, part 2 (NL, RLS)', '(December 3, 2014) part 1, part 2 (NL, RLS)', '(November 27, 2014) part 1, part 2 (NL)', '(November 26, 2014) part 1, part 2 (NL)', '(November 24, 2014) part 1, part 2 (NL, RLS)', '(November 20, 2014) part 1, part 2 (NL, RLS)', 'Bootleg (November 19, 2014) part 1, part 2 (NLS, RLS, JS!)', '(November 17, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(November 13, 2014) part 1, part 2 (NL, RLS)', "(November 12, 2014) part 1 Bootleg Nick's view part 2 (NL, RLS w/ rob, Baer)", 'Bootleg (November 6, 2014) part 1 part 2 (NL, RLS w/ rob, Baer)', '(November 5, 2014) part 1, part 2 (NL, RLS)', '(November 3, 2014) part 1, part 2 (NL, RLS w/ rob)', '(October 30, 2014) part 1, part 2 (NL, RLS)', '(October 29, 2014) part 1, part 2 (NL, RLS)', '(October 27, 2014) part 1, part 2 (NL, RLS w/rob, Baer)', 'Bootleg (October 23, 2014) part 1, part 2 (NL, RLS)', '(October 22, 2014) Part 1, Part 2 (NL, RLS w/ rob, MALF)', '(October 20, 2014) Part 1, Part 2 (NL, RLS w/ rob, Baer)', '(October 16, 2014) part 1, part 2 (NL, RLS)', '(October 15, 2014) part 1, part 2, part 3 (NL, RLS w/ rob, Baer)', '(October 13, 2014) part 1, part 2 (NL, RLS)', '(October 9, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(October 8, 2014) part 1, part 2 (NL, RLS)', '(October 6, 2014) part 1, part 2, part 3 (NL, RLS w/ rob)', '(October 2, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(October 1, 2014) part 1, part 2 (NL, RLS)', '(September 29, 2014) part 1, part 2 (NL)', '(September 25, 2014) part 1, part 2, part 3 (NL, RLS w/ rob, Mag)', '(September 24, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(September 22, 2014) part 1, part 2 (NL, RLS w/ rob, Mag)', '(September 18, 2014) part 1 part 2 (NL, RLS)', '(September 17, 2014) part 1 part 2 (NL, RLS w/ JS, rob)', '(September 15, 2014) part 1, part 2 (NL, RLS w/ JS, rob)', '(September 11, 2014) part 1, part 2 (NL, RLS)', '(September 10, 2014) part 1, part 2 (NL, RLS w/ JS, Mag)', '(September 8, 2014) part 1, Part 2 (NL, RLS w/ JS)', '(August 27, 2014) part 1, part 2 (NL, RLS in person)', '(August 25, 2014) part 1, part 2 (NL, RLS, Kate in person)', '(August 13, 2014) part 1, part 2 (NL, RLS)', '(August 12, 2014) part 1, part 2 (NL, RLS w/ rob)', '(August 7, 2014) part 1, part 2 (NL, RLS w/ rob)', '(August 6, 2014) part 1, part 2 (NL, RLS)', '(August 4, 2014) part 1, part 2 (NL, RLS w/ rob)', '(July 31, 2014) part 1, part 2 (NL, RLS)', '(July 30, 2014) part 1, part 2 (NL, RLS)', '(July 28, 2014) part 1, part 2 (NL, RLS w/ Kate, Baer, rob)', '(July 24, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(July 21, 2014) part 1, part 2 (NL, RLS)', '(July 16, 2014) part 1, part 2, part 3 (NL, RLS)', '(July 14, 2014) part 1, part 2 (NL, RLS)', '(July 10, 2014) part 1, part 2 (NL, RLS w/ Baer)', '(July 9, 2014) part 1, part 2 (NL, RLS)', '(July 7, 2014) part 1, part 2 (NL, RLS w/ Baer)', '(July 2, 2014) part 1, part 2 (NL, RLS)', '(June 30, 2014) part 1, part 2 (NL, RLS w/ Kate, rob)', '(June 26, 2014) part 1, part 2 (NL, RLS)', '(June 25, 2014) part 1, part 2 (NL, RLS)', '(June 19, 2014) part 1, part 2 (NL, RLS w/ rob)', '(June 18, 2014) part 1, part 2 (NL, RLS w/ Kate, rob, Baer, Mathas)', '(June 16, 2014) part 1, part 2 (NL, RLS, JS)', '(June 12, 2014) part 1, part 2 (NL, RLS, JS)', '(June 11, 2014) part 1, part 2 (NL, RLS, JS w/ Mathas)', '(June 9, 2014) part 1, part 2 (NL, RLS, JS)', '(June 5, 2014) part 1, part 2 (NL, RLS, JS)', '(June 4, 2014) part 1, part 2 (NL, RLS, JS)', '(June 2, 2014) part 1, part 2 (NL, RLS, JS)', '(May 29, 2014) part 1, part 2 (NL, RLS, JS)', '(May 28, 2014) part 1, part 2 (NL, RLS, JS)', '(May 26, 2014) part 1, part 2 (NL, RLS, JS)', '(May 15, 2014) part 1, part 2 (NL, RLS)', '(May 14, 2014) part 1, part 2 (NL, RLS)', '(May 12, 2014) part 1, part 2 (NL, RLS)', '(May 8, 2014) part 1, part 2 (NL, RLS, JS)', '(May 5, 2014) part 1, part 2 (NL, RLS, JS w/ Mike Bithell)', '(May 1, 2014) part 1, part 2 (NL, RLS, JS)', '(April 30, 2014) part 1, part 2 (NL, RLS, JS)', '(April 28, 2014) part 1, part 2 (NL, RLS, JS)', '(April 24, 2014) part 1, part 2 (NL, RLS, JS)', '(April 23, 2014) part 1, part 2 (NL, RLS, JS)', '(April 21, 2014) part 1, part 2 (NL, RLS, JS)', '(April 17, 2014) part 1, part 2 (NL, RLS, JS)', '(April 16, 2014) part 1, part 2 (NL, RLS, JS)', '(April 7, 2014) part 1, part 2 (NL, RLS, JS)', '(April 3, 2014) part 1, part 2 (NL, RLS, JS)', '(April 2, 2014) part 1, part 2 (NL, RLS, JS)', '(March 31, 2014) part 1, part 2 (NL, RLS, JS)', '(March 27, 2014) part 1, part 2 (NL, RLS, JS)', '(March 26, 2014) part 1, part 2 (NL, RLS, JS)', '(March 24, 2014) part 1, part 2 (NL, RLS, JS)', '(March 13, 2014) part 1, part 2 (NL, RLS, JS)', '(March 12, 2014) part 1, part 2 (NL, RLS, JS)', '(March 10, 2014) part 1, part 2 (NL, RLS, JS)', '(March 6, 2014) part 1, part 2 (NL, RLS, JS)', '(March 5, 2014) part 1, part 2 (NL, RLS, JS)', '(March 3, 2014) part 1, part 2 (NL, RLS, JS)', '(February 27, 2014) part 1, part 2 (NL, RLS, JS)', '(February 26, 2014) part 1, part 2 (NL, RLS, JS)', '(February 24, 2014) part 1, part 2 (NL, RLS, JS)', '(February 20, 2014) part 1, part 2 (NL, RLS)', '(February 19, 2014) part 1, part 2 (NL, RLS, JS)', '(February 17, 2014) part 1, part 2 (NL, RLS, JS)', '(February 13, 2014) part 1, part 2 (NL, RLS)', '(February 12, 2014) part 1, part 2, part 3, part 4, part 5 (NL, RLS)', '(February 10, 2014) part 1, part 2 (NL, RLS, JS)', '(February 6, 2014) part 1, part 2 (NL, RLS, JS)', '(February 5, 2014) part 1, part 2 (NL, RLS, JS, MALF)', '(February 3, 2014) part 1, part 2 (NL, RLS w/ rob, MALF)', '(January 30, 2014) part 1, part 2 (NL, RLS, JS w/ Mike Bithell)', '(January 29, 2014) part 1, part 2 (NL, RLS, JS)', '(January 27, 2014) (NL, RLS, JS w/ Crendor)', '(January 20, 2014) part 1, part 2 (NL, RLS, JS)', '(January 16, 2014) part 1, part 2 (NL, RLS, JS)', '(January 15, 2014) part 1, part 2 (NL, RLS, JS)', '(January 13, 2014) part 1, part 2 (NL, RLS, MALF)', '(January 9, 2014) part 1, part 2 (NL, RLS, MALF w/ rob)', '(January 8, 2014) part 1, part 2 (NL, RLS, JS)', '(January 6, 2014) part 1, part 2 (NL, RLS, JS)', '(December 19, 2013), part 1, part 2 (NL, RLS, JS)', '(December 18, 2013), part 1, part 2 (NL, JS)', '(December 16, 2013), part 1, part 2 (NL, RLS, JS)', '(December 12, 2013), part 1, part 2 (NL, RLS, JS)', '(December 11, 2013), part 1, part 2 (NL, RLS, JS)', '(December 9, 2013), part 1, part 2 (NL, RLS, JS)', '(December 5, 2013), part 1, part 2 (NL, RLS, JS)', '(December 4, 2013), part 1, part 2 (NL, RLS, JS)', '(December 2, 2013), part 1, part 2 (NL, RLS, JS)', '(November 28, 2013), part 1, part 2 (NL, JS)', '(November 27, 2013), part 1, part 2 (NL, RLS, JS)', '(November 25, 2013), part 1, Part 2 (NL, RLS, JS)', '(November 21, 2013), part 1, part 2 (NL, RLS, JS)', '(November 20, 2013), part 1, part 2 (NL, RLS, JS)', '(November 18, 2013), part 1, part 2 (NL, RLS, JS)', '(November 14, 2013), part 1, part 2 (NL, RLS, JS)', '(November 13, 2013), part 1, part 2 (NL, RLS, MALF)', '(November 11, 2013), part 1, part 2 (NL, RLS, MALF)', '(November 7, 2013), part 1, part 2 (NL, RLS, JS, MALF)', '(November 6, 2013), part 1, part 2 (NL, RLS, JS)', '(November 4, 2013), part 1, part 2 (NL, RLS, MALF)', '(October 31, 2013) part 1 part 2 (NL, RLS, JS)', '(October 30, 2013), part 1, part 2 (NL, RLS, JS)', '(October 28, 2013) (NL, RLS, JS)', '(October 24, 2013) Part 1, Part 2 (NL, RLS, JS)', '(October 23, 2013) (NL, RLS, JS w/ rob)', '(October 21, 2013) (NL, RLS, JS)', '(October 17, 2013) (NL, RLS, JS w/ rob)', '(October 16, 2013) (NL, RLS, JS w/ rob)', '(October 14, 2013) (NL, RLS, JS)', '(October 10, 2013) (NL, RLS, JS w/ RPG)', '(October 9, 2013) (NL, RLS, JS)', '(October 7, 2013) (NL, RLS, JS)', '(October 3, 2013) (NL, RLS, JS)', '(October 2, 2013) (NL, RLS, JS)', '(September 30, 2013) Part 1, Part 2 (NL, RLS, JS)', '(September 26, 2013) (NL, RLS, JS)', '(September 25, 2013) Part 1, Part 2 (NL, RLS, JS)', '(September 23, 2013) (NL, RLS, JS)', '(September 19, 2013) (NL, RLS, JS)', '(September 18, 2013) (NL, RLS, JS, MALF)', '(September 16, 2013) (NL, RLS, JS)', '(September 12, 2013) (NL, RLS, JS)', '(September 11, 2013) (NL, RLS, JS)', '(September 9, 2013) (NL, RLS, JS)', '(September 5, 2013) (NL, RLS, JS)', '(September 4, 2013) (NL, RLS, JS w/ Ohm)', '(August 26, 2013) part 1, part 2 (NL, RLS, JS)', '(August 22, 2013) (NL, RLS, JS w/ Ohm)', '(August 21, 2013) (NL, RLS, JS w/ Ohm)', '(August 19, 2013) (NL, RLS, JS)', '(August 15, 2013) (NL, RLS, JS)', '(August 14, 2013) (NL, RLS, JS)', '(August 12, 2013) (NL, RLS, JS)', '(August 1, 2013) (NL, RLS, JS w/ Kate)', '(July 31, 2013) (NL, RLS, JS w/ Kate)', '(July 29, 2013) (NL, RLS, JS w/ Ohm)', '(July 25, 2013) (NL, RLS, JS w/ Kate)', '(July 24, 2013) (NL, RLS, JS w/ Kate, MALF)', '(July 22, 2013) (NL, RLS, JS w/ Mike Bithell)', '(July 18, 2013) (NL, RLS, JS w/ Kate)', '(July 17, 2013) part 1, part 2 (NL, RLS, JS w/ Kate)', '(July 15, 2013) (NL, RLS, JS)', '(July 11, 2013) (NL, RLS, JS)', '(July 8, 2013) (NL, RLS, JS)', '(July 4, 2013) (NL, RLS, JS w/ Ohm)', '(July 3, 2013) (NL, RLS, JS w/ Kate, Ohm, rob)', '(July 1, 2013) (NL, RLS, JS)', '(June 20, 2013) (NL, RLS, JS w/ Ohm, rob, Pixel)', '(June 19, 2013) (NL, RLS, JS w/ Ohm, rob)', '(June 17, 2013) (NL, RLS, JS w/ Ohm, Green)', '(June 13, 2013) (NL, RLS, JS w/ Ohm, rob, LGW)', '(June 12, 2013) (NL, RLS, JS w/ Ohm, rob, RPG, Mathas)', '(June 10, 2013) (NL, RLS, JS w/ Ohm, rob)', '(June 5, 2013) (NL, RLS, JS w/ Ohm)', '(June 3, 2013) (NL, RLS, JS w/ Green, rob)', '(May 30, 2013) (NL, RLS, JS w/ Ohm, rob, Green)', '(May 29, 2013) (NL, RLS, JS)', '(May 27, 2013) (NL, RLS, JS w/ Ohm, rob)', '(May 23, 2013) (NL, RLS, JS w/ Kate, Ohm)', '(May 22, 2013) (NL, RLS, JS w/ Kate, Ohm, rob, Green)', '(May 20, 2013) (NL, RLS, JS w/ Kate, Ohm, Green)', '(May 16, 2013) (NL, RLS w/ Ohm, rob, Pixel)', '(May 15, 2013) (NL, RLS, Ohm)', '(May 13, 2013) (NL, RLS w/ Ohm, Mathas, rob)', '(May 2, 2013) (NL, RLS, JS w/ Ohm, RPG, Green)', '(May 1, 2013) (NL, RLS, JS w/ Ohm)', '(April 29, 2013) (NL, RLS, JS w/ Ohm, rob, Mathas, Green)', '(April 25, 2013) (NL, RLS, JS w/ rob, Green)', '(April 24, 2013) (NL, RLS, JS w/ Ohm)', '(April 22, 2013) (NL, RLS, JS w/ Ohm, rob, Green, RPG)', '(April 18, 2013) (NL, RLS w/ RPG, Green, Ohm, rob)', '(April 17, 2013) (NL, RLS, JS w/ Green, Ohm)', '(April 15, 2013) (NL, RLS, JS w/ Ohm, Green, rob, Mathas, MALF)', '(April 11, 2013) (NL, RLS, JS w/ Green, rob)', '(April 10, 2013) (NL, RLS, JS w/ Green, Ohm)', '(April 8, 2013) (NL, RLS, JS w/ Ohm, RPG, MALF)', '(April 4, 2013) (NL, RLS, JS w/ Green, Ohm)', '(April 3, 2013) (NL, RLS, JS w/ MALF, Ohm)', '(April 1, 2013) (NL, RLS, JS w/ Ohm)', '(March 28, 2013) (NL, RLS, JS w/ RPG, Ohm)', '(March 27, 2013) (NL, RLS, JS w/ Ohm)', '(March 18, 2013) (NL, RLS, JS)', '(March 14, 2013) (NL, RLS, JS w/ MALF)', '(March 13, 2013) (NL, RLS, JS w/ Ohm)', '(March 11, 2013) (NL, RLS, JS w/ Ohm)', '(March 6, 2013) (NL, RLS, JS)', '(March 4, 2013) (NL, RLS, JS)', '(February 28, 2013) (NL, RLS, JS)', '(February 27, 2013) (NL, Kate)', '(February 25, 2013) (NL)']
    

I'm going to use regex to split these up.


```python
import re
date = []
crew = []
for entry in date_crew:
    foo = re.search(r'\((.*)\)', entry).group(1)
    d = foo.split(r')')[0]
    date.append(d)
    c = foo.split(r'(')[-1]
    crew.append(c)
```

Now I'll start creating a data frame of this information


```python
date_df = pd.DataFrame(date, columns = ["Date"])
date_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>August 24, 2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August 23, 2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>August 21, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>August 17, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>August 16, 2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
games_df = pd.DataFrame(games, columns = ["Docket"])
games_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Docket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Passpartout, Party Panic, Pinturillo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Absolver, Golf It, Quiplash</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fire Pro Wrestling World, Ultimate Chicken Hor...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Geoguessr, Golf It, Quiplash</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nidhogg 2, Speedrunners, Pinturillo</td>
    </tr>
  </tbody>
</table>
</div>




```python
crew_df = pd.DataFrame(crew, columns = ["Crew"])
crew_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Crew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NL, RLS, CS, rob</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NL, RLS, rob w/ Baer, LGW, HCJ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NL, RLS, JS, rob</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NL w/ Sin, RLS, LGW, HCJ, Baer</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NL, RLS w/ rob, Baer, LGW, Dan</td>
    </tr>
  </tbody>
</table>
</div>



Now combine them


```python
nlss_df = pd.DataFrame()
nlss_df['Date'] = date_df['Date']
nlss_df['Crew'] = crew_df['Crew']
nlss_df['Docket'] = games_df['Docket']
nlss_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>August 24, 2017</td>
      <td>NL, RLS, CS, rob</td>
      <td>Passpartout, Party Panic, Pinturillo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August 23, 2017</td>
      <td>NL, RLS, rob w/ Baer, LGW, HCJ</td>
      <td>Absolver, Golf It, Quiplash</td>
    </tr>
    <tr>
      <th>2</th>
      <td>August 21, 2017</td>
      <td>NL, RLS, JS, rob</td>
      <td>Fire Pro Wrestling World, Ultimate Chicken Hor...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>August 17, 2017</td>
      <td>NL w/ Sin, RLS, LGW, HCJ, Baer</td>
      <td>Geoguessr, Golf It, Quiplash</td>
    </tr>
    <tr>
      <th>4</th>
      <td>August 16, 2017</td>
      <td>NL, RLS w/ rob, Baer, LGW, Dan</td>
      <td>Nidhogg 2, Speedrunners, Pinturillo</td>
    </tr>
  </tbody>
</table>
</div>




```python
nlss_df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>588</td>
      <td>588</td>
      <td>588</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>588</td>
      <td>223</td>
      <td>565</td>
    </tr>
    <tr>
      <th>top</th>
      <td>July 14, 2014</td>
      <td>NL, RLS, JS</td>
      <td>Dark Souls invasions, Trivia, Spelunky</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>111</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



I noticed that some lines had a link called "(continued)" in the games list. I want to get rid of these. While I'm at it, let's make the games docket contain the games as lists.


```python
improved = []
#For each docket
for d in nlss_df['Docket']:
    #Split docket into list of games
    d = d.split(r',')
    #For each game
    for g in d:
        #If game matches string to remove
        if g == r" (continued)" or g == r" (Continued)":
            #Remove game
            d.remove(g)
    improved.append(d)
nlss_df['Docket'] = improved
```


```python
nlss_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>August 24, 2017</td>
      <td>NL, RLS, CS, rob</td>
      <td>[Passpartout,  Party Panic,  Pinturillo]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August 23, 2017</td>
      <td>NL, RLS, rob w/ Baer, LGW, HCJ</td>
      <td>[Absolver,  Golf It,  Quiplash]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>August 21, 2017</td>
      <td>NL, RLS, JS, rob</td>
      <td>[Fire Pro Wrestling World,  Ultimate Chicken H...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>August 17, 2017</td>
      <td>NL w/ Sin, RLS, LGW, HCJ, Baer</td>
      <td>[Geoguessr,  Golf It,  Quiplash]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>August 16, 2017</td>
      <td>NL, RLS w/ rob, Baer, LGW, Dan</td>
      <td>[Nidhogg 2,  Speedrunners,  Pinturillo]</td>
    </tr>
  </tbody>
</table>
</div>




```python
nlss_df['Crew']
```




    0                                  NL, RLS, CS, rob
    1                    NL, RLS, rob w/ Baer, LGW, HCJ
    2                                  NL, RLS, JS, rob
    3                    NL w/ Sin, RLS, LGW, HCJ, Baer
    4                    NL, RLS w/ rob, Baer, LGW, Dan
    5                    NL, JS, MALF, LGW w/ Baer, HCJ
    6                                 NL, RLS, rob, LGW
    7                          NL, RLS, JS, rob w/ Baer
    8                          NL, RLS, CS w/ rob, MALF
    9                        NL, RLS, LGW w/ Baer, Kory
    10                NL, RLS, JS, rob w/ Sin, Baer, TB
    11                              NL, RLS, CS w/ MALF
    12                   NL, RLS w/ LGW, Sin, Baer, Dan
    13                               NL, RLS, JS w/ LGW
    14                         NL, RLS, CS, LGW w/ Baer
    15                  NL, RLS, LGW w/ MALF, rob, Baer
    16                        NL, RLS, rob w/ LGW, Baer
    17                             NL, RLS, rob w/ Baer
    18                          NL, RLS, JS w/ rob, Sin
    19                         NL, RLS, CS, rob w/ Baer
    20                        NL, RLS, rob w/ LGW, Baer
    21                   NL, RLS, rob w/ Sin, LGW, Baer
    22                           NL, RLS, CS, rob w/ JS
    23                        NL, RLS, rob w/ LGW, Baer
    24                   NL, RLS, JS w/ rob, Kate, Baer
    25                                  NL w/ MALF, rob
    26                        NL w/ MALF, LGW, JS, Baer
    27                                NL, RLS, JS, MALF
    28                       NL, RLS w/ rob, MALF, Baer
    29          NL, RLS, rob w/ Dan, Baer, Sin, Blueman
                               ...                     
    558                 NL, RLS, JS w/ Kate, Ohm, Green
    559                      NL, RLS w/ Ohm, rob, Pixel
    560                                    NL, RLS, Ohm
    561                     NL, RLS w/ Ohm, Mathas, rob
    562                  NL, RLS, JS w/ Ohm, RPG, Green
    563                              NL, RLS, JS w/ Ohm
    564          NL, RLS, JS w/ Ohm, rob, Mathas, Green
    565                       NL, RLS, JS w/ rob, Green
    566                              NL, RLS, JS w/ Ohm
    567             NL, RLS, JS w/ Ohm, rob, Green, RPG
    568                 NL, RLS w/ RPG, Green, Ohm, rob
    569                       NL, RLS, JS w/ Green, Ohm
    570    NL, RLS, JS w/ Ohm, Green, rob, Mathas, MALF
    571                       NL, RLS, JS w/ Green, rob
    572                       NL, RLS, JS w/ Green, Ohm
    573                   NL, RLS, JS w/ Ohm, RPG, MALF
    574                       NL, RLS, JS w/ Green, Ohm
    575                        NL, RLS, JS w/ MALF, Ohm
    576                              NL, RLS, JS w/ Ohm
    577                         NL, RLS, JS w/ RPG, Ohm
    578                              NL, RLS, JS w/ Ohm
    579                                     NL, RLS, JS
    580                             NL, RLS, JS w/ MALF
    581                              NL, RLS, JS w/ Ohm
    582                              NL, RLS, JS w/ Ohm
    583                                     NL, RLS, JS
    584                                     NL, RLS, JS
    585                                     NL, RLS, JS
    586                                        NL, Kate
    587                                              NL
    Name: Crew, Length: 588, dtype: object



I want to split on "w/" so each crew member is individual item. I'm also going to put them into a list.


```python
improved = []
#For each cast of crew
for e in nlss_df['Crew']:
    #Split cast into list of members
    e = e.split(r',')
    #For each member
    for m in e:
        #If member contains a /w
        if r'w/' in m:
            both = m.split(r'w/')
            e.remove(m)
            e.extend(both)
    improved.append(e)
improved[:20]
```




    [['NL', ' RLS', ' CS', ' rob'],
     ['NL', ' RLS', ' LGW', ' HCJ', ' rob ', ' Baer'],
     ['NL', ' RLS', ' JS', ' rob'],
     [' RLS', ' LGW', ' HCJ', ' Baer', 'NL ', ' Sin'],
     ['NL', ' Baer', ' LGW', ' Dan', ' RLS ', ' rob'],
     ['NL', ' JS', ' MALF', ' HCJ', ' LGW ', ' Baer'],
     ['NL', ' RLS', ' rob', ' LGW'],
     ['NL', ' RLS', ' JS', ' rob ', ' Baer'],
     ['NL', ' RLS', ' MALF', ' CS ', ' rob'],
     ['NL', ' RLS', ' Kory', ' LGW ', ' Baer'],
     ['NL', ' RLS', ' JS', ' Baer', ' TB', ' rob ', ' Sin'],
     ['NL', ' RLS', ' CS ', ' MALF'],
     ['NL', ' Sin', ' Baer', ' Dan', ' RLS ', ' LGW'],
     ['NL', ' RLS', ' JS ', ' LGW'],
     ['NL', ' RLS', ' CS', ' LGW ', ' Baer'],
     ['NL', ' RLS', ' rob', ' Baer', ' LGW ', ' MALF'],
     ['NL', ' RLS', ' Baer', ' rob ', ' LGW'],
     ['NL', ' RLS', ' rob ', ' Baer'],
     ['NL', ' RLS', ' Sin', ' JS ', ' rob'],
     ['NL', ' RLS', ' CS', ' rob ', ' Baer']]



Strip extra spaces


```python
fullstripped = []
for entry in improved:
    stripped = []
    for member in entry:
        member = member.strip(' ')
        stripped.append(member)
    fullstripped.append(stripped)
fullstripped[:10]
```




    [['NL', 'RLS', 'CS', 'rob'],
     ['NL', 'RLS', 'LGW', 'HCJ', 'rob', 'Baer'],
     ['NL', 'RLS', 'JS', 'rob'],
     ['RLS', 'LGW', 'HCJ', 'Baer', 'NL', 'Sin'],
     ['NL', 'Baer', 'LGW', 'Dan', 'RLS', 'rob'],
     ['NL', 'JS', 'MALF', 'HCJ', 'LGW', 'Baer'],
     ['NL', 'RLS', 'rob', 'LGW'],
     ['NL', 'RLS', 'JS', 'rob', 'Baer'],
     ['NL', 'RLS', 'MALF', 'CS', 'rob'],
     ['NL', 'RLS', 'Kory', 'LGW', 'Baer']]



Let's make the names consistant. Luckily I know the aliases that are used. Let's see what we're working with.


```python
names = []
for entry in fullstripped:
    for user in entry:
        if user not in names:
            names.append(user)
print(names)
```

    ['NL', 'RLS', 'CS', 'rob', 'LGW', 'HCJ', 'Baer', 'JS', 'Sin', 'Dan', 'MALF', 'Kory', 'TB', 'Kate', 'Blueman', 'BaerBaer', 'Mathas', 'Crendor', 'BRex', 'GhostBill', 'alpacapatrol', 'dan', '', 'Brex', 'Fox', 'Arumba', 'baer', 'cobaltstreak', 'fox', 'Mag', 'NLS', 'JS!', 'RLS in person', 'Kate in person', 'Mike Bithell', 'RPG', 'Ohm', 'Pixel', 'Green']
    

Translated: Northernlion, RockLeeSmile, CobaltStreak, AlpacaPatrol, LastGreyWolf, HCJustin, BaerTaffy, JSmithOTI, Sinvicta, DanGheesling, MALF, FlackBlag, TotalBiscuit, LovelyMomo, Blueman, BaerTaffy, MathasGames, Crendor, BananasaurusRex, NOTREAL, AlpacaPatrol, DanGheesling, BananasaurusRex, MALF, Arumba, BaerTaffy, CobaltStreak, MALF, Magresta, Northernlion, JSmithOTI, RockLeeSmile, LovelyMomo, MikeBithell, RedPandaGamer, OhmWrecker, PrescriptionPixel, Green9090


```python
foo = "Northernlion, RockLeeSmile, CobaltStreak, AlpacaPatrol, LastGreyWolf, HCJustin, BaerTaffy, JSmithOTI, Sinvicta, DanGheesling, MALF, FlackBlag, TotalBiscuit, LovelyMomo, Blueman, BaerTaffy, MathasGames, Crendor, BananasaurusRex, NOTREAL, AlpacaPatrol, DanGheesling, NOTREAL, BananasaurusRex, MALF, Arumba, BaerTaffy, CobaltStreak, MALF, Magresta, Northernlion, JSmithOTI, RockLeeSmile, LovelyMomo, MikeBithell, RedPandaGamer, OhmWrecker, PrescriptionPixel, Green9090"
translated = foo.split(", ")
translated
```




    ['Northernlion',
     'RockLeeSmile',
     'CobaltStreak',
     'AlpacaPatrol',
     'LastGreyWolf',
     'HCJustin',
     'BaerTaffy',
     'JSmithOTI',
     'Sinvicta',
     'DanGheesling',
     'MALF',
     'FlackBlag',
     'TotalBiscuit',
     'LovelyMomo',
     'Blueman',
     'BaerTaffy',
     'MathasGames',
     'Crendor',
     'BananasaurusRex',
     'NOTREAL',
     'AlpacaPatrol',
     'DanGheesling',
     'NOTREAL',
     'BananasaurusRex',
     'MALF',
     'Arumba',
     'BaerTaffy',
     'CobaltStreak',
     'MALF',
     'Magresta',
     'Northernlion',
     'JSmithOTI',
     'RockLeeSmile',
     'LovelyMomo',
     'MikeBithell',
     'RedPandaGamer',
     'OhmWrecker',
     'PrescriptionPixel',
     'Green9090']




```python
guests = []
for cast in fullstripped:
    guests.append([translated[names.index(user)] for user in cast])
```


```python
#Replace first names with second names
guests[0]
```




    ['Northernlion', 'RockLeeSmile', 'CobaltStreak', 'AlpacaPatrol']



Looking better. Let's swap it into our DF.


```python
nlss_df['Crew'] = guests
nlss_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>August 24, 2017</td>
      <td>[Northernlion, RockLeeSmile, CobaltStreak, Alp...</td>
      <td>[Passpartout,  Party Panic,  Pinturillo]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August 23, 2017</td>
      <td>[Northernlion, RockLeeSmile, LastGreyWolf, HCJ...</td>
      <td>[Absolver,  Golf It,  Quiplash]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>August 21, 2017</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI, Alpaca...</td>
      <td>[Fire Pro Wrestling World,  Ultimate Chicken H...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>August 17, 2017</td>
      <td>[RockLeeSmile, LastGreyWolf, HCJustin, BaerTaf...</td>
      <td>[Geoguessr,  Golf It,  Quiplash]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>August 16, 2017</td>
      <td>[Northernlion, BaerTaffy, LastGreyWolf, DanGhe...</td>
      <td>[Nidhogg 2,  Speedrunners,  Pinturillo]</td>
    </tr>
  </tbody>
</table>
</div>



# Adding more stats

File from https://sullygnome.com/channel/Northernlion/365/streams

This version can only go back 365 days. Can we create a column for date that matches nlss_df format? If so, we can combine overlapping stats. I also have a larger CSV which I'm working on in FullCSV.ipynb. I will combine these once formated correctly.


```python
import os
import glob
print(os.getcwd())

allFiles = glob.glob(r"data\*.csv")
stream_df = pd.DataFrame()
l = []
for foo in allFiles:
    stream_df = pd.read_csv(foo,index_col=None, header=0)
    l.append(stream_df)
stream_df = pd.concat(l)

#stream_df = pd.read_csv(r'StreamStats365.csv')
stream_df
```

    C:\Users\sonofabrat\Documents\Data_Science\NLSS_Project
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
      <th>Games</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Thursday 21st September 2017 21:15</td>
      <td>Friday 22nd September 2017 00:15</td>
      <td>13</td>
      <td>4636</td>
      <td>5390</td>
      <td>36</td>
      <td>2.77</td>
      <td>5984</td>
      <td>460.31</td>
      <td>GeoGuessr|GeoGuessr|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Wednesday 20th September 2017 21:15</td>
      <td>Thursday 21st September 2017 00:00</td>
      <td>12</td>
      <td>5245</td>
      <td>5988</td>
      <td>49</td>
      <td>4.08</td>
      <td>4881</td>
      <td>406.75</td>
      <td>GeoGuessr|GeoGuessr|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Tuesday 19th September 2017 20:45</td>
      <td>Tuesday 19th September 2017 23:15</td>
      <td>11</td>
      <td>2718</td>
      <td>3319</td>
      <td>48</td>
      <td>4.36</td>
      <td>2160</td>
      <td>196.36</td>
      <td>Life Is Strange|Life_Is_Strange|https://static...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Monday 18th September 2017 22:15</td>
      <td>Tuesday 19th September 2017 01:15</td>
      <td>13</td>
      <td>4911</td>
      <td>5325</td>
      <td>45</td>
      <td>3.46</td>
      <td>4793</td>
      <td>368.69</td>
      <td>Ben and Ed: Blood Party|Ben_and_Ed_Blood_Party...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Sunday 17th September 2017 20:30</td>
      <td>Sunday 17th September 2017 23:30</td>
      <td>13</td>
      <td>3074</td>
      <td>3527</td>
      <td>97</td>
      <td>7.46</td>
      <td>3380</td>
      <td>260.00</td>
      <td>Fallout 3|Fallout_3|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Thursday 14th September 2017 21:15</td>
      <td>Friday 15th September 2017 00:15</td>
      <td>13</td>
      <td>4562</td>
      <td>5363</td>
      <td>51</td>
      <td>3.92</td>
      <td>4818</td>
      <td>370.62</td>
      <td>The Binding of Isaac: Afterbirth|The_Binding_o...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Wednesday 13th September 2017 21:15</td>
      <td>Thursday 14th September 2017 00:15</td>
      <td>13</td>
      <td>4596</td>
      <td>5601</td>
      <td>48</td>
      <td>3.69</td>
      <td>6174</td>
      <td>474.92</td>
      <td>Tooth and Tail|Tooth_and_Tail|https://static-c...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Tuesday 12th September 2017 20:15</td>
      <td>Tuesday 12th September 2017 23:00</td>
      <td>12</td>
      <td>2557</td>
      <td>2980</td>
      <td>52</td>
      <td>4.33</td>
      <td>3881</td>
      <td>323.42</td>
      <td>Life Is Strange|Life_Is_Strange|https://static...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Monday 11th September 2017 22:00</td>
      <td>Tuesday 12th September 2017 01:15</td>
      <td>14</td>
      <td>4683</td>
      <td>5555</td>
      <td>89</td>
      <td>6.36</td>
      <td>3478</td>
      <td>248.43</td>
      <td>Rock of Ages II: Bigger &amp; Boulder|Rock_of_Ages...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Sunday 10th September 2017 20:30</td>
      <td>Sunday 10th September 2017 23:30</td>
      <td>13</td>
      <td>3304</td>
      <td>3604</td>
      <td>174</td>
      <td>13.38</td>
      <td>5416</td>
      <td>416.62</td>
      <td>Fallout 3|Fallout_3|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Thursday 7th September 2017 21:15</td>
      <td>Friday 8th September 2017 00:30</td>
      <td>14</td>
      <td>4498</td>
      <td>5639</td>
      <td>81</td>
      <td>5.79</td>
      <td>6323</td>
      <td>451.64</td>
      <td>GeoGuessr|GeoGuessr|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Wednesday 6th September 2017 21:15</td>
      <td>Thursday 7th September 2017 00:15</td>
      <td>13</td>
      <td>5153</td>
      <td>5872</td>
      <td>110</td>
      <td>8.46</td>
      <td>5880</td>
      <td>452.31</td>
      <td>The Binding of Isaac: Afterbirth|The_Binding_o...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Wednesday 6th September 2017 00:45</td>
      <td>Wednesday 6th September 2017 04:00</td>
      <td>14</td>
      <td>2731</td>
      <td>3039</td>
      <td>31</td>
      <td>2.21</td>
      <td>2102</td>
      <td>150.14</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS|PLAYERUNKNOWNS_B...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>Wednesday 30th August 2017 21:15</td>
      <td>Thursday 31st August 2017 00:15</td>
      <td>13</td>
      <td>3334</td>
      <td>4977</td>
      <td>23</td>
      <td>1.77</td>
      <td>1818</td>
      <td>139.85</td>
      <td>Tower Unite|Tower_Unite|https://static-cdn.jtv...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>Tuesday 29th August 2017 20:15</td>
      <td>Tuesday 29th August 2017 23:30</td>
      <td>14</td>
      <td>2554</td>
      <td>3062</td>
      <td>55</td>
      <td>3.93</td>
      <td>3943</td>
      <td>281.64</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS|PLAYERUNKNOWNS_B...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>Monday 28th August 2017 22:15</td>
      <td>Tuesday 29th August 2017 01:15</td>
      <td>13</td>
      <td>5767</td>
      <td>6405</td>
      <td>140</td>
      <td>10.77</td>
      <td>5398</td>
      <td>415.23</td>
      <td>Golf It!|Golf_It_|https://static-cdn.jtvnw.net...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>Sunday 27th August 2017 20:15</td>
      <td>Sunday 27th August 2017 23:15</td>
      <td>13</td>
      <td>3857</td>
      <td>4295</td>
      <td>265</td>
      <td>20.38</td>
      <td>7595</td>
      <td>584.23</td>
      <td>Fallout 3|Fallout_3|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>Thursday 24th August 2017 21:15</td>
      <td>Friday 25th August 2017 00:15</td>
      <td>13</td>
      <td>4715</td>
      <td>5751</td>
      <td>91</td>
      <td>7.00</td>
      <td>5400</td>
      <td>415.38</td>
      <td>Passpartout: The Starving Artist|Passpartout_T...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>Wednesday 23rd August 2017 21:15</td>
      <td>Thursday 24th August 2017 00:15</td>
      <td>13</td>
      <td>5214</td>
      <td>6185</td>
      <td>170</td>
      <td>13.08</td>
      <td>5962</td>
      <td>458.62</td>
      <td>Super Blood Hockey|Super_Blood_Hockey|https://...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>Tuesday 22nd August 2017 20:30</td>
      <td>Tuesday 22nd August 2017 23:45</td>
      <td>14</td>
      <td>3576</td>
      <td>4089</td>
      <td>133</td>
      <td>9.50</td>
      <td>6584</td>
      <td>470.29</td>
      <td>Absolver|Absolver|https://static-cdn.jtvnw.net...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>Monday 21st August 2017 22:15</td>
      <td>Tuesday 22nd August 2017 01:15</td>
      <td>13</td>
      <td>4797</td>
      <td>5347</td>
      <td>95</td>
      <td>7.31</td>
      <td>5065</td>
      <td>389.62</td>
      <td>Fire Pro Wrestling World|Fire_Pro_Wrestling_Wo...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>Sunday 20th August 2017 20:15</td>
      <td>Sunday 20th August 2017 23:15</td>
      <td>13</td>
      <td>3920</td>
      <td>4388</td>
      <td>325</td>
      <td>25.00</td>
      <td>4417</td>
      <td>339.77</td>
      <td>Fallout 3|Fallout_3|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>Thursday 17th August 2017 21:15</td>
      <td>Friday 18th August 2017 00:15</td>
      <td>13</td>
      <td>5214</td>
      <td>6075</td>
      <td>108</td>
      <td>8.31</td>
      <td>5365</td>
      <td>412.69</td>
      <td>GeoGuessr|GeoGuessr|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>Wednesday 16th August 2017 21:15</td>
      <td>Thursday 17th August 2017 00:30</td>
      <td>14</td>
      <td>4681</td>
      <td>5250</td>
      <td>116</td>
      <td>8.29</td>
      <td>5823</td>
      <td>415.93</td>
      <td>Nidhogg II|Nidhogg_II|https://static-cdn.jtvnw...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>Tuesday 15th August 2017 20:15</td>
      <td>Tuesday 15th August 2017 23:15</td>
      <td>13</td>
      <td>2514</td>
      <td>3343</td>
      <td>84</td>
      <td>6.46</td>
      <td>3798</td>
      <td>292.15</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS|PLAYERUNKNOWNS_B...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>Monday 14th August 2017 22:15</td>
      <td>Tuesday 15th August 2017 01:15</td>
      <td>13</td>
      <td>5294</td>
      <td>6094</td>
      <td>156</td>
      <td>12.00</td>
      <td>4971</td>
      <td>382.38</td>
      <td>HITMAN|HITMAN|https://static-cdn.jtvnw.net/ttv...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>Sunday 13th August 2017 20:30</td>
      <td>Sunday 13th August 2017 23:45</td>
      <td>14</td>
      <td>4821</td>
      <td>5714</td>
      <td>429</td>
      <td>30.64</td>
      <td>4866</td>
      <td>347.57</td>
      <td>Fallout 3|Fallout_3|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>Friday 11th August 2017 20:15</td>
      <td>Friday 11th August 2017 23:15</td>
      <td>13</td>
      <td>2706</td>
      <td>3351</td>
      <td>73</td>
      <td>5.62</td>
      <td>3784</td>
      <td>291.08</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS|PLAYERUNKNOWNS_B...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>Thursday 10th August 2017 21:15</td>
      <td>Friday 11th August 2017 00:15</td>
      <td>13</td>
      <td>4374</td>
      <td>5301</td>
      <td>99</td>
      <td>7.62</td>
      <td>5089</td>
      <td>391.46</td>
      <td>Ultimate Chicken Horse|Ultimate_Chicken_Horse|...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>Monday 7th August 2017 22:15</td>
      <td>Tuesday 8th August 2017 01:15</td>
      <td>13</td>
      <td>5453</td>
      <td>6025</td>
      <td>189</td>
      <td>14.54</td>
      <td>6552</td>
      <td>504.00</td>
      <td>Passpartout: The Starving Artist|Passpartout_T...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>187</td>
      <td>Monday 14th November 2016 23:15</td>
      <td>Tuesday 15th November 2016 02:00</td>
      <td>12</td>
      <td>5955</td>
      <td>7472</td>
      <td>49</td>
      <td>4.08</td>
      <td>10333</td>
      <td>861.08</td>
      <td>Rocket League|Rocket_League|https://static-cdn...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>188</td>
      <td>Sunday 13th November 2016 21:15</td>
      <td>Monday 14th November 2016 00:00</td>
      <td>12</td>
      <td>4159</td>
      <td>4811</td>
      <td>71</td>
      <td>5.92</td>
      <td>6177</td>
      <td>514.75</td>
      <td>HITMAN|HITMAN|https://static-cdn.jtvnw.net/ttv...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>189</td>
      <td>Thursday 10th November 2016 23:15</td>
      <td>Friday 11th November 2016 02:15</td>
      <td>13</td>
      <td>5388</td>
      <td>6059</td>
      <td>64</td>
      <td>4.92</td>
      <td>8787</td>
      <td>675.92</td>
      <td>Pictionary|Pictionary|https://static-cdn.jtvnw...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>190</td>
      <td>Wednesday 9th November 2016 22:15</td>
      <td>Thursday 10th November 2016 01:15</td>
      <td>13</td>
      <td>6090</td>
      <td>6825</td>
      <td>78</td>
      <td>6.00</td>
      <td>16664</td>
      <td>1281.85</td>
      <td>Quiplash|Quiplash|https://static-cdn.jtvnw.net...</td>
    </tr>
    <tr>
      <th>89</th>
      <td>191</td>
      <td>Monday 7th November 2016 23:15</td>
      <td>Tuesday 8th November 2016 02:00</td>
      <td>12</td>
      <td>5620</td>
      <td>6676</td>
      <td>40</td>
      <td>3.33</td>
      <td>5980</td>
      <td>498.33</td>
      <td>Duck Game|Duck_Game|https://static-cdn.jtvnw.n...</td>
    </tr>
    <tr>
      <th>90</th>
      <td>192</td>
      <td>Sunday 6th November 2016 21:15</td>
      <td>Monday 7th November 2016 00:15</td>
      <td>13</td>
      <td>4030</td>
      <td>4657</td>
      <td>49</td>
      <td>3.77</td>
      <td>6728</td>
      <td>517.54</td>
      <td>HITMAN|HITMAN|https://static-cdn.jtvnw.net/ttv...</td>
    </tr>
    <tr>
      <th>91</th>
      <td>193</td>
      <td>Thursday 3rd November 2016 22:15</td>
      <td>Friday 4th November 2016 01:30</td>
      <td>14</td>
      <td>5453</td>
      <td>6035</td>
      <td>32</td>
      <td>2.29</td>
      <td>7300</td>
      <td>521.43</td>
      <td>Gang Beasts|Gang_Beasts|https://static-cdn.jtv...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>194</td>
      <td>Wednesday 2nd November 2016 21:45</td>
      <td>Thursday 3rd November 2016 00:15</td>
      <td>11</td>
      <td>6433</td>
      <td>7430</td>
      <td>55</td>
      <td>5.00</td>
      <td>5674</td>
      <td>515.82</td>
      <td>The Jackbox Party Pack 3|The_Jackbox_Party_Pac...</td>
    </tr>
    <tr>
      <th>93</th>
      <td>195</td>
      <td>Monday 31st October 2016 22:15</td>
      <td>Tuesday 1st November 2016 01:00</td>
      <td>12</td>
      <td>4749</td>
      <td>5680</td>
      <td>40</td>
      <td>3.33</td>
      <td>8185</td>
      <td>682.08</td>
      <td>The Culling|The_Culling|https://static-cdn.jtv...</td>
    </tr>
    <tr>
      <th>94</th>
      <td>196</td>
      <td>Sunday 30th October 2016 20:15</td>
      <td>Sunday 30th October 2016 23:45</td>
      <td>15</td>
      <td>3361</td>
      <td>4008</td>
      <td>41</td>
      <td>2.73</td>
      <td>6710</td>
      <td>447.33</td>
      <td>Hearthstone|Hearthstone_Heroes_of_Warcraft|htt...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>197</td>
      <td>Thursday 27th October 2016 22:15</td>
      <td>Friday 28th October 2016 01:15</td>
      <td>13</td>
      <td>5144</td>
      <td>5829</td>
      <td>38</td>
      <td>2.92</td>
      <td>6161</td>
      <td>473.92</td>
      <td>London 2012 - The Official Video Game of the O...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>198</td>
      <td>Wednesday 26th October 2016 21:15</td>
      <td>Thursday 27th October 2016 00:00</td>
      <td>12</td>
      <td>5460</td>
      <td>6448</td>
      <td>50</td>
      <td>4.17</td>
      <td>7945</td>
      <td>662.08</td>
      <td>The Jackbox Party Pack 3|The_Jackbox_Party_Pac...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>199</td>
      <td>Monday 24th October 2016 22:15</td>
      <td>Tuesday 25th October 2016 01:00</td>
      <td>12</td>
      <td>5916</td>
      <td>6611</td>
      <td>48</td>
      <td>4.00</td>
      <td>6637</td>
      <td>553.08</td>
      <td>Keep Talking and Nobody Explodes|Keep_Talking_...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>200</td>
      <td>Sunday 23rd October 2016 20:15</td>
      <td>Sunday 23rd October 2016 23:30</td>
      <td>14</td>
      <td>3158</td>
      <td>3688</td>
      <td>48</td>
      <td>3.43</td>
      <td>4293</td>
      <td>306.64</td>
      <td>Grand Theft Auto V|Grand_Theft_Auto_V|https://...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>201</td>
      <td>Thursday 20th October 2016 22:15</td>
      <td>Friday 21st October 2016 01:15</td>
      <td>13</td>
      <td>5611</td>
      <td>6751</td>
      <td>58</td>
      <td>4.46</td>
      <td>8468</td>
      <td>651.38</td>
      <td>The Jackbox Party Pack 3|The_Jackbox_Party_Pac...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>202</td>
      <td>Wednesday 19th October 2016 21:15</td>
      <td>Thursday 20th October 2016 00:15</td>
      <td>13</td>
      <td>4877</td>
      <td>6492</td>
      <td>50</td>
      <td>3.85</td>
      <td>8711</td>
      <td>670.08</td>
      <td>The Jackbox Party Pack 3|The_Jackbox_Party_Pac...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>203</td>
      <td>Tuesday 18th October 2016 20:15</td>
      <td>Tuesday 18th October 2016 23:15</td>
      <td>13</td>
      <td>2310</td>
      <td>2779</td>
      <td>26</td>
      <td>2.00</td>
      <td>3713</td>
      <td>285.62</td>
      <td>Grand Theft Auto V|Grand_Theft_Auto_V|https://...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>204</td>
      <td>Monday 17th October 2016 22:15</td>
      <td>Tuesday 18th October 2016 01:15</td>
      <td>13</td>
      <td>6386</td>
      <td>7224</td>
      <td>94</td>
      <td>7.23</td>
      <td>8313</td>
      <td>639.46</td>
      <td>The Jackbox Party Pack 3|The_Jackbox_Party_Pac...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>205</td>
      <td>Thursday 13th October 2016 18:15</td>
      <td>Thursday 13th October 2016 21:15</td>
      <td>13</td>
      <td>3377</td>
      <td>3819</td>
      <td>25</td>
      <td>1.92</td>
      <td>4766</td>
      <td>366.62</td>
      <td>Worms W.M.D|Worms_W.M.D|https://static-cdn.jtv...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>206</td>
      <td>Wednesday 12th October 2016 21:15</td>
      <td>Thursday 13th October 2016 00:15</td>
      <td>13</td>
      <td>4033</td>
      <td>4546</td>
      <td>27</td>
      <td>2.08</td>
      <td>5776</td>
      <td>444.31</td>
      <td>Quiplash|Quiplash|https://static-cdn.jtvnw.net...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>207</td>
      <td>Monday 10th October 2016 22:15</td>
      <td>Tuesday 11th October 2016 01:15</td>
      <td>13</td>
      <td>5144</td>
      <td>5710</td>
      <td>78</td>
      <td>6.00</td>
      <td>8049</td>
      <td>619.15</td>
      <td>Shadow Warrior 2|Shadow_Warrior_2|https://stat...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>208</td>
      <td>Sunday 9th October 2016 20:15</td>
      <td>Sunday 9th October 2016 23:30</td>
      <td>14</td>
      <td>2905</td>
      <td>3220</td>
      <td>40</td>
      <td>2.86</td>
      <td>4484</td>
      <td>320.29</td>
      <td>Grand Theft Auto V|Grand_Theft_Auto_V|https://...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>209</td>
      <td>Thursday 6th October 2016 22:15</td>
      <td>Friday 7th October 2016 01:15</td>
      <td>13</td>
      <td>4478</td>
      <td>5344</td>
      <td>36</td>
      <td>2.77</td>
      <td>7618</td>
      <td>586.00</td>
      <td>Pictionary|Pictionary|https://static-cdn.jtvnw...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>210</td>
      <td>Wednesday 5th October 2016 21:15</td>
      <td>Thursday 6th October 2016 00:00</td>
      <td>12</td>
      <td>5189</td>
      <td>5940</td>
      <td>36</td>
      <td>3.00</td>
      <td>6791</td>
      <td>565.92</td>
      <td>Quiplash|Quiplash|https://static-cdn.jtvnw.net...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>211</td>
      <td>Monday 3rd October 2016 22:15</td>
      <td>Tuesday 4th October 2016 01:15</td>
      <td>13</td>
      <td>5315</td>
      <td>5805</td>
      <td>52</td>
      <td>4.00</td>
      <td>6637</td>
      <td>510.54</td>
      <td>Ultimate Chicken Horse|Ultimate_Chicken_Horse|...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>212</td>
      <td>Sunday 2nd October 2016 20:15</td>
      <td>Sunday 2nd October 2016 23:15</td>
      <td>13</td>
      <td>3470</td>
      <td>3923</td>
      <td>63</td>
      <td>4.85</td>
      <td>5338</td>
      <td>410.62</td>
      <td>Grand Theft Auto V|Grand_Theft_Auto_V|https://...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>213</td>
      <td>Thursday 29th September 2016 22:15</td>
      <td>Friday 30th September 2016 01:00</td>
      <td>12</td>
      <td>5277</td>
      <td>6388</td>
      <td>80</td>
      <td>6.67</td>
      <td>6481</td>
      <td>540.08</td>
      <td>ShellShock Live|ShellShock_Live|https://static...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>214</td>
      <td>Wednesday 28th September 2016 21:15</td>
      <td>Wednesday 28th September 2016 23:45</td>
      <td>11</td>
      <td>4692</td>
      <td>5284</td>
      <td>35</td>
      <td>3.18</td>
      <td>5040</td>
      <td>458.18</td>
      <td>Golf With Your Friends|Golf_With_Your_Friends|...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>215</td>
      <td>Monday 26th September 2016 22:00</td>
      <td>Tuesday 27th September 2016 01:00</td>
      <td>13</td>
      <td>5152</td>
      <td>6164</td>
      <td>38</td>
      <td>2.92</td>
      <td>5823</td>
      <td>447.92</td>
      <td>Tricky Towers|Tricky_Towers|https://static-cdn...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>216</td>
      <td>Sunday 25th September 2016 20:15</td>
      <td>Sunday 25th September 2016 23:15</td>
      <td>13</td>
      <td>3322</td>
      <td>3666</td>
      <td>72</td>
      <td>5.54</td>
      <td>6456</td>
      <td>496.62</td>
      <td>Grand Theft Auto V|Grand_Theft_Auto_V|https://...</td>
    </tr>
  </tbody>
</table>
<p>215 rows × 11 columns</p>
</div>




```python
formatted = []
order = [1,0,2]
for date in stream_df['Stream start time']:
    dmy = date.split(' ')[1:-1] #Date/Month/Year
    dmy[0] = dmy[0][:-2] #Remove day suffixes
    mdy = [dmy[i] for i in order]
    formatted.append(str(mdy[0] + " " + mdy[1] + ", " + mdy[2]))
formatted[:15]
```




    ['September 21, 2017',
     'September 20, 2017',
     'September 19, 2017',
     'September 18, 2017',
     'September 17, 2017',
     'September 14, 2017',
     'September 13, 2017',
     'September 12, 2017',
     'September 11, 2017',
     'September 10, 2017',
     'September 7, 2017',
     'September 6, 2017',
     'September 6, 2017',
     'August 30, 2017',
     'August 29, 2017']




```python
stream_df["Date"] = formatted
stream_df = stream_df.reset_index()
stream_df.index = stream_df["index"]
stream_df = stream_df.drop('index', axis=1)
stream_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
      <th>Games</th>
      <th>Date</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Thursday 21st September 2017 21:15</td>
      <td>Friday 22nd September 2017 00:15</td>
      <td>13</td>
      <td>4636</td>
      <td>5390</td>
      <td>36</td>
      <td>2.77</td>
      <td>5984</td>
      <td>460.31</td>
      <td>GeoGuessr|GeoGuessr|https://static-cdn.jtvnw.n...</td>
      <td>September 21, 2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Wednesday 20th September 2017 21:15</td>
      <td>Thursday 21st September 2017 00:00</td>
      <td>12</td>
      <td>5245</td>
      <td>5988</td>
      <td>49</td>
      <td>4.08</td>
      <td>4881</td>
      <td>406.75</td>
      <td>GeoGuessr|GeoGuessr|https://static-cdn.jtvnw.n...</td>
      <td>September 20, 2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Tuesday 19th September 2017 20:45</td>
      <td>Tuesday 19th September 2017 23:15</td>
      <td>11</td>
      <td>2718</td>
      <td>3319</td>
      <td>48</td>
      <td>4.36</td>
      <td>2160</td>
      <td>196.36</td>
      <td>Life Is Strange|Life_Is_Strange|https://static...</td>
      <td>September 19, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Monday 18th September 2017 22:15</td>
      <td>Tuesday 19th September 2017 01:15</td>
      <td>13</td>
      <td>4911</td>
      <td>5325</td>
      <td>45</td>
      <td>3.46</td>
      <td>4793</td>
      <td>368.69</td>
      <td>Ben and Ed: Blood Party|Ben_and_Ed_Blood_Party...</td>
      <td>September 18, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Sunday 17th September 2017 20:30</td>
      <td>Sunday 17th September 2017 23:30</td>
      <td>13</td>
      <td>3074</td>
      <td>3527</td>
      <td>97</td>
      <td>7.46</td>
      <td>3380</td>
      <td>260.00</td>
      <td>Fallout 3|Fallout_3|https://static-cdn.jtvnw.n...</td>
      <td>September 17, 2017</td>
    </tr>
  </tbody>
</table>
</div>



There was a day where an extra non-NLSS stream happened. It messes up our ordering so let's remove it.


```python
stream_df[stream_df["Date"]=='January 4, 2017']
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
      <th>Games</th>
      <th>Date</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53</th>
      <td>154</td>
      <td>Wednesday 4th January 2017 22:15</td>
      <td>Thursday 5th January 2017 01:00</td>
      <td>12</td>
      <td>7779</td>
      <td>9636</td>
      <td>172</td>
      <td>14.33</td>
      <td>17681</td>
      <td>1473.42</td>
      <td>Pictionary|Pictionary|https://static-cdn.jtvnw...</td>
      <td>January 4, 2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
stream_df = stream_df[stream_df['Unnamed: 0'] != 0]
stream_df[stream_df["Date"]=='January 4, 2017']
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
      <th>Games</th>
      <th>Date</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>53</th>
      <td>154</td>
      <td>Wednesday 4th January 2017 22:15</td>
      <td>Thursday 5th January 2017 01:00</td>
      <td>12</td>
      <td>7779</td>
      <td>9636</td>
      <td>172</td>
      <td>14.33</td>
      <td>17681</td>
      <td>1473.42</td>
      <td>Pictionary|Pictionary|https://static-cdn.jtvnw...</td>
      <td>January 4, 2017</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined = nlss_df.merge(stream_df)
#drop eronious columns
combined = combined.drop('Games', 1)
combined = combined.drop('Unnamed: 0', 1)
```


```python
nlss_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>August 24, 2017</td>
      <td>[Northernlion, RockLeeSmile, CobaltStreak, Alp...</td>
      <td>[Passpartout,  Party Panic,  Pinturillo]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August 23, 2017</td>
      <td>[Northernlion, RockLeeSmile, LastGreyWolf, HCJ...</td>
      <td>[Absolver,  Golf It,  Quiplash]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>August 21, 2017</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI, Alpaca...</td>
      <td>[Fire Pro Wrestling World,  Ultimate Chicken H...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>August 17, 2017</td>
      <td>[RockLeeSmile, LastGreyWolf, HCJustin, BaerTaf...</td>
      <td>[Geoguessr,  Golf It,  Quiplash]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>August 16, 2017</td>
      <td>[Northernlion, BaerTaffy, LastGreyWolf, DanGhe...</td>
      <td>[Nidhogg 2,  Speedrunners,  Pinturillo]</td>
    </tr>
  </tbody>
</table>
</div>




```python
combined.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>August 24, 2017</td>
      <td>[Northernlion, RockLeeSmile, CobaltStreak, Alp...</td>
      <td>[Passpartout,  Party Panic,  Pinturillo]</td>
      <td>Thursday 24th August 2017 21:15</td>
      <td>Friday 25th August 2017 00:15</td>
      <td>13</td>
      <td>4715</td>
      <td>5751</td>
      <td>91</td>
      <td>7.00</td>
      <td>5400</td>
      <td>415.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August 23, 2017</td>
      <td>[Northernlion, RockLeeSmile, LastGreyWolf, HCJ...</td>
      <td>[Absolver,  Golf It,  Quiplash]</td>
      <td>Wednesday 23rd August 2017 21:15</td>
      <td>Thursday 24th August 2017 00:15</td>
      <td>13</td>
      <td>5214</td>
      <td>6185</td>
      <td>170</td>
      <td>13.08</td>
      <td>5962</td>
      <td>458.62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>August 21, 2017</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI, Alpaca...</td>
      <td>[Fire Pro Wrestling World,  Ultimate Chicken H...</td>
      <td>Monday 21st August 2017 22:15</td>
      <td>Tuesday 22nd August 2017 01:15</td>
      <td>13</td>
      <td>4797</td>
      <td>5347</td>
      <td>95</td>
      <td>7.31</td>
      <td>5065</td>
      <td>389.62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>August 17, 2017</td>
      <td>[RockLeeSmile, LastGreyWolf, HCJustin, BaerTaf...</td>
      <td>[Geoguessr,  Golf It,  Quiplash]</td>
      <td>Thursday 17th August 2017 21:15</td>
      <td>Friday 18th August 2017 00:15</td>
      <td>13</td>
      <td>5214</td>
      <td>6075</td>
      <td>108</td>
      <td>8.31</td>
      <td>5365</td>
      <td>412.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>August 16, 2017</td>
      <td>[Northernlion, BaerTaffy, LastGreyWolf, DanGhe...</td>
      <td>[Nidhogg 2,  Speedrunners,  Pinturillo]</td>
      <td>Wednesday 16th August 2017 21:15</td>
      <td>Thursday 17th August 2017 00:30</td>
      <td>14</td>
      <td>4681</td>
      <td>5250</td>
      <td>116</td>
      <td>8.29</td>
      <td>5823</td>
      <td>415.93</td>
    </tr>
  </tbody>
</table>
</div>




```python
result = pd.concat([nlss_df, combined], axis=1)
#Removes repeat columns
result = result.T.groupby(level=0).first().T
#Reorder columns
result = result[['Date','Crew','Docket','Stream start time','End time','Stream length','Avg Viewers','Peak viewers','Followers gained','Followers per hour','Views','Views per hour']]
```


```python
nlss_df = result
nlss_df.loc[50]
```




    Date                                                     April 19, 2017
    Crew                  [Northernlion, RockLeeSmile, BaerTaffy, LastGr...
    Docket                               [Afterbirth+,  Golf It,  Quiplash]
    Stream start time                       Wednesday 19th April 2017 21:15
    End time                                 Thursday 20th April 2017 00:15
    Stream length                                                        13
    Avg Viewers                                                        5594
    Peak viewers                                                       6733
    Followers gained                                                     61
    Followers per hour                                                 4.69
    Views                                                              9790
    Views per hour                                                   753.08
    Name: 50, dtype: object




```python
nlss_df[70:85]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70</th>
      <td>February 15, 2017</td>
      <td>[Northernlion, RockLeeSmile, AlpacaPatrol, Las...</td>
      <td>[Ultimate Chicken Horse,  Golf It,  For Honor]</td>
      <td>Wednesday 15th February 2017 22:15</td>
      <td>Thursday 16th February 2017 01:15</td>
      <td>13</td>
      <td>5306</td>
      <td>6401</td>
      <td>36</td>
      <td>2.77</td>
      <td>7061</td>
      <td>543.15</td>
    </tr>
    <tr>
      <th>71</th>
      <td>February 13, 2017</td>
      <td>[Northernlion, LastGreyWolf, JSmithOTI, Alpaca...</td>
      <td>[Afterbirth+,  Scribblenauts,  Invisigun Heroes]</td>
      <td>Monday 13th February 2017 23:15</td>
      <td>Tuesday 14th February 2017 02:15</td>
      <td>13</td>
      <td>6760</td>
      <td>8271</td>
      <td>51</td>
      <td>3.92</td>
      <td>8006</td>
      <td>615.85</td>
    </tr>
    <tr>
      <th>72</th>
      <td>February 9, 2017</td>
      <td>[Northernlion, RockLeeSmile, AlpacaPatrol, Bae...</td>
      <td>[Afterbirth+,  For Honor,  Quiplash]</td>
      <td>Thursday 9th February 2017 23:15</td>
      <td>Friday 10th February 2017 02:15</td>
      <td>13</td>
      <td>6210</td>
      <td>7351</td>
      <td>41</td>
      <td>3.15</td>
      <td>7083</td>
      <td>544.85</td>
    </tr>
    <tr>
      <th>73</th>
      <td>February 8, 2017</td>
      <td>[Northernlion, RockLeeSmile, LastGreyWolf, MAL...</td>
      <td>[Afterbirth+,  Sonic All Star Racing Transform...</td>
      <td>Wednesday 8th February 2017 22:15</td>
      <td>Thursday 9th February 2017 01:30</td>
      <td>14</td>
      <td>5521</td>
      <td>6280</td>
      <td>102</td>
      <td>7.29</td>
      <td>13499</td>
      <td>964.21</td>
    </tr>
    <tr>
      <th>74</th>
      <td>February 6, 2017</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI, MALF]</td>
      <td>[Afterbirth+,  Disc Jam,  Gang Beasts]</td>
      <td>Monday 6th February 2017 23:15</td>
      <td>Tuesday 7th February 2017 02:15</td>
      <td>13</td>
      <td>6515</td>
      <td>7876</td>
      <td>82</td>
      <td>6.31</td>
      <td>10047</td>
      <td>772.85</td>
    </tr>
    <tr>
      <th>75</th>
      <td>February 2, 2017</td>
      <td>[Northernlion, RockLeeSmile, AlpacaPatrol, MAL...</td>
      <td>[Afterbirth+,  Sonic All Stars Racing Transfor...</td>
      <td>Thursday 2nd February 2017 23:15</td>
      <td>Friday 3rd February 2017 02:15</td>
      <td>13</td>
      <td>6095</td>
      <td>6954</td>
      <td>75</td>
      <td>5.77</td>
      <td>8344</td>
      <td>641.85</td>
    </tr>
    <tr>
      <th>76</th>
      <td>February 1, 2017</td>
      <td>[Northernlion, RockLeeSmile, AlpacaPatrol, Las...</td>
      <td>[Afterbirth+,  Ultimate Chicken Horse,  Golf W...</td>
      <td>Wednesday 1st February 2017 22:15</td>
      <td>Thursday 2nd February 2017 01:00</td>
      <td>12</td>
      <td>6062</td>
      <td>6565</td>
      <td>73</td>
      <td>6.08</td>
      <td>9477</td>
      <td>789.75</td>
    </tr>
    <tr>
      <th>77</th>
      <td>January 31, 2017</td>
      <td>[Northernlion, JSmithOTI, MALF, AlpacaPatrol, ...</td>
      <td>[Afterbirth+,  Scribblenauts,  Quiplash]</td>
      <td>Tuesday 31st January 2017 00:15</td>
      <td>Tuesday 31st January 2017 03:00</td>
      <td>12</td>
      <td>7445</td>
      <td>8393</td>
      <td>78</td>
      <td>6.5</td>
      <td>7689</td>
      <td>640.75</td>
    </tr>
    <tr>
      <th>78</th>
      <td>January 26, 2017</td>
      <td>[Northernlion, MALF, AlpacaPatrol, LastGreyWolf]</td>
      <td>[Enter the Gungeon,  Disc Jam,  Pinturillo]</td>
      <td>Thursday 26th January 2017 23:15</td>
      <td>Friday 27th January 2017 02:00</td>
      <td>12</td>
      <td>5804</td>
      <td>6611</td>
      <td>36</td>
      <td>3</td>
      <td>5792</td>
      <td>482.67</td>
    </tr>
    <tr>
      <th>79</th>
      <td>January 25, 2017</td>
      <td>[Northernlion, AlpacaPatrol, DanGheesling, Sin...</td>
      <td>[Afterbirth+,  Ultimate Chicken Horse,  Quiplash]</td>
      <td>Wednesday 25th January 2017 22:15</td>
      <td>Thursday 26th January 2017 01:00</td>
      <td>12</td>
      <td>6772</td>
      <td>7702</td>
      <td>54</td>
      <td>4.5</td>
      <td>6710</td>
      <td>559.17</td>
    </tr>
    <tr>
      <th>80</th>
      <td>January 23, 2017</td>
      <td>[Northernlion, JSmithOTI, MALF]</td>
      <td>[Afterbirth+,  Scribblenauts attempt,  Who Wan...</td>
      <td>Monday 23rd January 2017 23:15</td>
      <td>Tuesday 24th January 2017 01:00</td>
      <td>11</td>
      <td>7750</td>
      <td>9012</td>
      <td>92</td>
      <td>8.36</td>
      <td>12865</td>
      <td>1169.55</td>
    </tr>
    <tr>
      <th>81</th>
      <td>January 19, 2017</td>
      <td>[Northernlion, MALF, AlpacaPatrol, LastGreyWolf]</td>
      <td>[Afterbirth+,  Who Wants to be a Millionaire, ...</td>
      <td>Thursday 19th January 2017 23:15</td>
      <td>Friday 20th January 2017 02:00</td>
      <td>12</td>
      <td>7618</td>
      <td>9180</td>
      <td>119</td>
      <td>9.92</td>
      <td>10207</td>
      <td>850.58</td>
    </tr>
    <tr>
      <th>82</th>
      <td>January 18, 2017</td>
      <td>[Northernlion, AlpacaPatrol, LastGreyWolf, MALF]</td>
      <td>[Afterbirth+,  Ultimate Chicken Horse,  Pintur...</td>
      <td>Wednesday 18th January 2017 22:15</td>
      <td>Thursday 19th January 2017 01:15</td>
      <td>13</td>
      <td>6695</td>
      <td>7419</td>
      <td>129</td>
      <td>9.92</td>
      <td>11058</td>
      <td>850.62</td>
    </tr>
    <tr>
      <th>83</th>
      <td>January 16, 2017</td>
      <td>[Northernlion, JSmithOTI, AlpacaPatrol, BaerTa...</td>
      <td>[Afterbirth+,  Disc Jam,  Trivia Murder Party,...</td>
      <td>Monday 16th January 2017 23:15</td>
      <td>Tuesday 17th January 2017 02:15</td>
      <td>13</td>
      <td>8204</td>
      <td>9526</td>
      <td>158</td>
      <td>12.15</td>
      <td>14836</td>
      <td>1141.23</td>
    </tr>
    <tr>
      <th>84</th>
      <td>January 12, 2017</td>
      <td>[Northernlion, RockLeeSmile, Crendor, MALF, La...</td>
      <td>[Afterbirth+,  Dead By Daylight,  Quiplash]</td>
      <td>Thursday 12th January 2017 23:15</td>
      <td>Friday 13th January 2017 02:15</td>
      <td>13</td>
      <td>7193</td>
      <td>7856</td>
      <td>108</td>
      <td>8.31</td>
      <td>11611</td>
      <td>893.15</td>
    </tr>
  </tbody>
</table>
</div>




```python
nlss_df.loc[nlss_df['Date']=="Wednesday 8th February 2017 22:15"]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



# Let's Explore

Our stats have been compiled. Now let's look around. Which show had most peak viewers?


```python
mpv = nlss_df.loc[nlss_df['Peak viewers'].idxmax()]
print("Date:", mpv["Date"])
print("Peak viewers:", mpv['Peak viewers'])
print("Peak percentage:", (mpv['Peak viewers']/mpv['Views'])*100)
print("Total viewers:", mpv['Views'])
print("Games:", mpv['Docket'])
nlss_df.loc[nlss_df['Peak viewers'].idxmax()]
```

    Date: December 6, 2016
    Peak viewers: 9752.0
    Peak percentage: 81.37516688918558
    Total viewers: 11984.0
    Games: ['Ultimate Chicken Horse', ' Move or Die', ' Quiplash']
    




    Date                                                   December 6, 2016
    Crew                  [Northernlion, RockLeeSmile, MALF, LastGreyWol...
    Docket                [Ultimate Chicken Horse,  Move or Die,  Quiplash]
    Stream start time                       Tuesday 6th December 2016 23:15
    End time                              Wednesday 7th December 2016 02:00
    Stream length                                                        12
    Avg Viewers                                                        5748
    Peak viewers                                                       9752
    Followers gained                                                     72
    Followers per hour                                                    6
    Views                                                             11984
    Views per hour                                                   998.67
    Name: 101, dtype: object




```python
nlss_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>August 24, 2017</td>
      <td>[Northernlion, RockLeeSmile, CobaltStreak, Alp...</td>
      <td>[Passpartout,  Party Panic,  Pinturillo]</td>
      <td>Thursday 24th August 2017 21:15</td>
      <td>Friday 25th August 2017 00:15</td>
      <td>13</td>
      <td>4715</td>
      <td>5751</td>
      <td>91</td>
      <td>7</td>
      <td>5400</td>
      <td>415.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August 23, 2017</td>
      <td>[Northernlion, RockLeeSmile, LastGreyWolf, HCJ...</td>
      <td>[Absolver,  Golf It,  Quiplash]</td>
      <td>Wednesday 23rd August 2017 21:15</td>
      <td>Thursday 24th August 2017 00:15</td>
      <td>13</td>
      <td>5214</td>
      <td>6185</td>
      <td>170</td>
      <td>13.08</td>
      <td>5962</td>
      <td>458.62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>August 21, 2017</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI, Alpaca...</td>
      <td>[Fire Pro Wrestling World,  Ultimate Chicken H...</td>
      <td>Monday 21st August 2017 22:15</td>
      <td>Tuesday 22nd August 2017 01:15</td>
      <td>13</td>
      <td>4797</td>
      <td>5347</td>
      <td>95</td>
      <td>7.31</td>
      <td>5065</td>
      <td>389.62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>August 17, 2017</td>
      <td>[RockLeeSmile, LastGreyWolf, HCJustin, BaerTaf...</td>
      <td>[Geoguessr,  Golf It,  Quiplash]</td>
      <td>Thursday 17th August 2017 21:15</td>
      <td>Friday 18th August 2017 00:15</td>
      <td>13</td>
      <td>5214</td>
      <td>6075</td>
      <td>108</td>
      <td>8.31</td>
      <td>5365</td>
      <td>412.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>August 16, 2017</td>
      <td>[Northernlion, BaerTaffy, LastGreyWolf, DanGhe...</td>
      <td>[Nidhogg 2,  Speedrunners,  Pinturillo]</td>
      <td>Wednesday 16th August 2017 21:15</td>
      <td>Thursday 17th August 2017 00:30</td>
      <td>14</td>
      <td>4681</td>
      <td>5250</td>
      <td>116</td>
      <td>8.29</td>
      <td>5823</td>
      <td>415.93</td>
    </tr>
  </tbody>
</table>
</div>



# NLSS Dataframe


```python
len(nlss_df)
nlss_df.head()
nlss_df.tail()
```




    588






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>August 24, 2017</td>
      <td>[Northernlion, RockLeeSmile, CobaltStreak, Alp...</td>
      <td>[Passpartout,  Party Panic,  Pinturillo]</td>
      <td>Thursday 24th August 2017 21:15</td>
      <td>Friday 25th August 2017 00:15</td>
      <td>13</td>
      <td>4715</td>
      <td>5751</td>
      <td>91</td>
      <td>7</td>
      <td>5400</td>
      <td>415.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August 23, 2017</td>
      <td>[Northernlion, RockLeeSmile, LastGreyWolf, HCJ...</td>
      <td>[Absolver,  Golf It,  Quiplash]</td>
      <td>Wednesday 23rd August 2017 21:15</td>
      <td>Thursday 24th August 2017 00:15</td>
      <td>13</td>
      <td>5214</td>
      <td>6185</td>
      <td>170</td>
      <td>13.08</td>
      <td>5962</td>
      <td>458.62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>August 21, 2017</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI, Alpaca...</td>
      <td>[Fire Pro Wrestling World,  Ultimate Chicken H...</td>
      <td>Monday 21st August 2017 22:15</td>
      <td>Tuesday 22nd August 2017 01:15</td>
      <td>13</td>
      <td>4797</td>
      <td>5347</td>
      <td>95</td>
      <td>7.31</td>
      <td>5065</td>
      <td>389.62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>August 17, 2017</td>
      <td>[RockLeeSmile, LastGreyWolf, HCJustin, BaerTaf...</td>
      <td>[Geoguessr,  Golf It,  Quiplash]</td>
      <td>Thursday 17th August 2017 21:15</td>
      <td>Friday 18th August 2017 00:15</td>
      <td>13</td>
      <td>5214</td>
      <td>6075</td>
      <td>108</td>
      <td>8.31</td>
      <td>5365</td>
      <td>412.69</td>
    </tr>
    <tr>
      <th>4</th>
      <td>August 16, 2017</td>
      <td>[Northernlion, BaerTaffy, LastGreyWolf, DanGhe...</td>
      <td>[Nidhogg 2,  Speedrunners,  Pinturillo]</td>
      <td>Wednesday 16th August 2017 21:15</td>
      <td>Thursday 17th August 2017 00:30</td>
      <td>14</td>
      <td>4681</td>
      <td>5250</td>
      <td>116</td>
      <td>8.29</td>
      <td>5823</td>
      <td>415.93</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Crew</th>
      <th>Docket</th>
      <th>Stream start time</th>
      <th>End time</th>
      <th>Stream length</th>
      <th>Avg Viewers</th>
      <th>Peak viewers</th>
      <th>Followers gained</th>
      <th>Followers per hour</th>
      <th>Views</th>
      <th>Views per hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>583</th>
      <td>March 6, 2013</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI]</td>
      <td>[Dark Souls invasions,  Trivia,  Tomb Raider, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>584</th>
      <td>March 4, 2013</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI]</td>
      <td>[Delver's Drop with Ryan Baker and Ryan Burrel...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>585</th>
      <td>February 28, 2013</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI]</td>
      <td>[Dark Souls,  Trivia,  Trials Evolution,  Ask ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>586</th>
      <td>February 27, 2013</td>
      <td>[Northernlion, LovelyMomo]</td>
      <td>[Dark Souls,  Trivia,  More Dark Souls,  Ask m...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>587</th>
      <td>February 25, 2013</td>
      <td>[Northernlion]</td>
      <td>[Runner 2,  Super House of Dead Ninjas,  Trivi...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



8/25/2017 - 2/25/2013

# Current Goals

I'm working on an ipynb called FullCSV. I got this by contacting the owner of Sullygnome.com. This CSV goes back further than the ones I've been working with and so will be helpful to use. However, there are differences in how it is formated compared to the CSVs used here. I'm currently working to set the dates up in the same format so I can add in the stats from FullCSV not found here.

My new major issue is rather recent. Twitch recently (earlier this week as of writing) [changed their API](https://blog.twitch.tv/the-new-twitch-api-be3fb2b078e6), breaking most existing apps using it. This new update requires new types of authenication. I'm working on learning the new API, however my formerly working Twitch comment downloader is not in working order yet. As far as I can tell, none of the comment downloaders I found on Github currently work, but a few developers are working on updating them to the newest API.

I converted a file of Twitch emote commands from Twitch's API site. I will attempt to find user channel specific emote commands and combine it with list. I can then use this master list to search for or omit emotes from analysis.

# Sharability of Data

I've been in contact with the people who created the NLSS Docket list and the CSVs. They both gave me permission to use and publicly release the data in my project. I will work on organizing a cleaned up folder of my data to publish on Github.
