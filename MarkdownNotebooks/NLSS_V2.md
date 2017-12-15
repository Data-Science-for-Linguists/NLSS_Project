
# Overview

This is an updated form that takes the place of the original NLSS notebook. However, it uses the more complete CSV files instead of the three smaller CSVs used in the original. In this Notebook, I organize data from a text document of dockets. The dockets include the date of the stream, who was on the show, and what games were played. I split up the data into categories and make lists out of the participants and games. I also take data on stream statistics from a CSV file. The data from the dockets and the stream stats are put together into one dataframe.


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

# Cleaning up dataframe


```python
#Open up the docket and take a look
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




```python
#Take out unneeded text
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




```python
#Create a list of games
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




```python
#Create a list of crew members and show dates
date_crew = []
for s in shows:
    dc = s.split('\n')[0]
    date_crew.append(dc)
print(date_crew)
```

    ['(August 24, 2017) (NL, RLS, CS, rob)', '(August 23, 2017) (NL, RLS, rob w/ Baer, LGW, HCJ)', '(August 21, 2017) (NL, RLS, JS, rob)', '(August 17, 2017) (NL w/ Sin, RLS, LGW, HCJ, Baer)', '(August 16, 2017) (NL, RLS w/ rob, Baer, LGW, Dan)', '(August 14, 2017) (NL, JS, MALF, LGW w/ Baer, HCJ)', '(August 10, 2017) (NL, RLS, rob, LGW)', '(August 7, 2017) (NL, RLS, JS, rob w/ Baer)', '(August 3, 2017) (NL, RLS, CS w/ rob, MALF)', '(August 2, 2017) (NL, RLS, LGW w/ Baer, Kory)', '(July 31, 2017) (NL, RLS, JS, rob w/ Sin, Baer, TB)', '(July 27, 2017) (NL, RLS, CS w/ MALF)', '(July 26, 2017) (NL, RLS w/ LGW, Sin, Baer, Dan)', '(July 24, 2017) (NL, RLS, JS w/ LGW)', '(July 20, 2017) (NL, RLS, CS, LGW w/ Baer)', '(July 19, 2017) (NL, RLS, LGW w/ MALF, rob, Baer)', '(July 13, 2017) (NL, RLS, rob w/ LGW, Baer)', '(July 12, 2017) (NL, RLS, rob w/ Baer)', '(July 10, 2017) part 1, part 2 (NL, RLS, JS w/ rob, Sin)', '(July 6, 2017) (NL, RLS, CS, rob w/ Baer)', '(July 5, 2017) (NL, RLS, rob w/ LGW, Baer)', '(July 3, 2017) (NL, RLS, rob w/ Sin, LGW, Baer)', '(June 22, 2017) (NL, RLS, CS, rob w/ JS)', '(June 21, 2017) Nick view (NL, RLS, rob w/ LGW, Baer)', '(June 19, 2017) Nick view (NL, RLS, JS w/ rob, Kate, Baer)', 'Solo (June 15, 2017) (NL w/ MALF, rob)', 'Solo (June 14, 2017) (NL w/ MALF, LGW, JS, Baer)', 'NLSS Masters (June 12, 2017) (NL, RLS, JS, MALF)', '(June 8, 2017) (NL, RLS w/ rob, MALF, Baer)', '(June 7, 2017) (NL, RLS, rob w/ Dan, Baer, Sin, Blueman)', '(June 5, 2017) Nick view (NL, RLS, JS w/ MALF)', '(June 1, 2017) Nick view (NL, RLS, CS w/ rob, Baer)', '(May 31, 2017) (NL, rob, LGW, Baer)', '(May 29, 2017) (NL, RLS, JS w/ rob, MALF)', '(May 25, 2017) (NL, RLS, rob w/ MALF, Baer)', '(May 24, 2017) (NL, RLS, rob, LGW w/ MALF, Dan)', '(May 22, 2017) (NL, RLS, JS w/ MALF)', '(May 18, 2017) (NL, RLS, CS, rob w/ Kate, Baer)', '(May 17, 2017) (NL, RLS w/ Dan, LGW)', '(May 15, 2017) (NL, RLS, JS w/ rob, LGW, MALF)', '(May 11, 2017) (NL, RLS, rob, Baer w/ LGW, Sin)', '(May 10, 2017) (NL, RLS, rob w/ Baer, LGW, Dan)', '(May 8, 2017) (NL, RLS w/ rob, MALF)', '(May 4, 2017) (NL, RLS, rob w/ MALF, Baer)', '(May 3, 2017) (NL, RLS, rob w/ MALF)', '(May 1, 2017) (NL, RLS, JS w/ rob, MALF)', '(April 27, 2017) (NL, RLS w/ JS, MALF, Baer, LGW, Kate)', '(April 26, 2017) (NL, RLS, rob, LGW w/ Baer)', '(April 24, 2017) (NL, RLS, rob, LGW w/ Baer)', '(April 20, 2017) Nick view (NL, RLS, JS, CS w/ rob)', '(April 19, 2017) Nick view (NL, RLS, LGW w/ rob, Baer)', '(April 6, 2017) part 1 part 2 Nick view (NL, RLS, CS w/ LGW)', '(April 5, 2017) (NL, RLS, rob w/ Baer)', '(April 3, 2017) (NL, JS w/ MALF, LGW, rob, Baer, Sin)', '(March 30, 2017) (NL, RLS, CS w/ BaerBaer, MALF)', '(March 29, 2017) Nick Video (NL, RLS, rob, LGW w/ Dan)', '(March 27, 2017) (NL, RLS, JS w/ rob)', '(March 23, 2017) (NL, RLS, CS w/ MALF)', '(March 22, 2017) Nick Video (NL, RLS, LGW w/ MALF, Baer)', '(March 20, 2017) (NL, RLS, LGW w/ Baer, rob)', '(March 16, 2017) (NL, CS, LGW w/ MALF, Baer)', '(March 15, 2017) (NL, RLS, LGW w/ Kate)', '(March 8, 2017) (NL, RLS, rob w/ LGW, Baer, Sin)', '(March 6, 2017) (NL, RLS, JS w/ LGW, Baer, Sin)', '(March 1, 2017) (NL, rob, LGW w/ MALF, Baer)', '(February 27, 2017) (NL, RLS, JS w/ rob, LGW, Baer)', '(February 23, 2017) (NL, RLS, MALF w/ LGW, Baer)', '(February 22, 2017) (NL, RLS, rob, LGW w/ Dan)', '(February 20, 2017) Nick view (NL, RLS, JS, LGW w/ rob, Baer, Sin)', '(February 16, 2017) (NL, RLS, LGW w/ rob, Baer, MALF)', '(February 15, 2017) (NL, RLS, rob, LGW w/ Mathas)', '(February 13, 2017) (NL, JS w/ rob, LGW)', '(February 9, 2017) (NL, RLS, rob, LGW w/ MALF, Baer)', '(February 8, 2017) part 1 part 2 (NL, RLS, CS w/ rob, LGW, MALF, Baer)', '(February 6, 2017) (NL, RLS, JS w/ MALF)', '(February 2, 2017) (NL, RLS, rob, LGW w/ Baer, MALF)', '(February 1, 2017) (NL, RLS, rob, LGW)', '(January 31, 2017) (NL, JS, MALF, rob w/ Sin)', '(January 26, 2017) (NL, MALF, rob, LGW)', '(January 25, 2017) (NL, rob, LGW w/ MALF, Dan, Sin, Kate)', '(January 23, 2017) part 1 part 2 (NL, JS, MALF)', '(January 19, 2017) (NL, MALF, rob, LGW)', '(January 18, 2017) (NL, rob, LGW w/ MALF)', '(January 16, 2017) part 1 part 2 (NL, JS, MALF w/ LGW, rob, Baer, Sin, Dan)', '(January 12, 2017) (NL, RLS, MALF w/ LGW, Crendor)', '(January 11, 2017) (NL, RLS w/ rob, LGW, Dan)', '(January 9, 2017) (NL, RLS, JS w/ Baer, rob, LGW)', '(January 5, 2017) (NL, RLS, rob w/ MALF, Baer, rob)', '(January 4, 2017) part 1 part 2 (NL, RLS, rob w/ MALF, LGW)', '(January 2, 2017) (NL, RLS, JS w/ LGW)', '(December 29, 2016) (NL, RLS, LGW w/ rob, Baer, JS)', '(December 28, 2016) (NL, RLS, LGW w/ rob, Baer)', '(December 26, 2016) (NL, RLS, JS w/ rob, Sin)', 'Bootleg (December 22, 2016) part 1 part 2 part 3 (NL, LGW w/ Kate)', '(December 21, 2016) (NL, RLS, CS w/ Baer, rob, LGW, Dan)', '(December 19, 2016) (NL, RLS, JS, rob w/ LGW)', '(December 15, 2016) (NL, RLS, MALF, rob w/ Kate, LGW)', '(December 14, 2016) (NL, RLS, CS w/ LGW)', '(December 12, 2016) (NL, RLS, JS, LGW)', '(December 8, 2016) (NL, RLS, MALF, rob w/ LGW)', '(December 7, 2016) (NL, RLS, rob, LGW w/ MALF)', '(December 6, 2016) (NL, RLS, MALF, LGW w/ rob)', '(December 5, 2016) (NL, RLS, JS, rob w/ LGW, Baer)', '(November 30, 2016) (NL, RLS, rob, LGW)', '(November 28, 2016) (NL, RLS, rob, LGW w/ Baer)', '(November 24, 2016) (NL, MALF, rob, LGW)', '(November 23, 2016) (NL, RLS, CS w/ rob, LGW, Dan, MALF, Baer)', '(November 21, 2016) (NL, RLS, rob, LGW w/ Baer, Sin)', '(November 17, 2016) (NL, RLS, MALF, rob w/ Baer, LGW)', '(November 16, 2016) (NL, RLS, CS w/ Mathas, rob)', '(November 14, 2016) (NL, RLS, JS, rob w/ LGW, BRex)', '(November 10, 2016) (NL, RLS, rob, LGW w/ JS, MALF)', '(November 9, 2016) (NL, JS, MALF, rob, LGW)', '(November 7, 2016) (NL, RLS, JS, LGW)', '(November 3, 2016) (NL, RLS w/ rob, LGW)', '(November 2, 2016) Nick view (NL, RLS w/ MALF, rob, LGW, Dan)', '(October 31, 2016) (NL, RLS, JS, Sin w/ rob, LGW)', '(October 27, 2016) (NL, RLS, MALF w/ LGW, rob)', '(October 26, 2016) Nick view (NL, RLS, CS, rob w/ LGW)', '(October 24, 2016) (NL, RLS, JS, LGW w/ rob, Baer)', '(October 20, 2016) (NL, RLS, MALF w/ rob, Baer, LGW)', '(October 19, 2016)(NL, RLS, CS, rob w/ Baer)', '(October 17, 2016) (NL, RLS, JS, MALF w/ rob, LGW, Baer)', '(October 13, 2016) (NL, RLS w/ rob, LGW)', '(October 12, 2016) (NL, RLS, CS, LGW w/ MALF, rob, Baer)', '(October 10, 2016) (NL, RLS, JS, rob w/ LGW, GhostBill)', '(October 6, 2016) (NL, RLS, MALF, LGW w/ rob, Baer)', '(October 5, 2016) (NL, RLS, rob, LGW w/ Baer, Sin)', '(October 3, 2016) (NL, RLS, JS w/ rob)', '(September 29, 2016) (NL, RLS, rob, LGW)', '(September 28, 2016) (NL, RLS, CS w/ rob, LGW)', '(September 26, 2016) (NL, RLS, JS w/ rob, Kate)', '(September 22, 2016) (NL, RLS, MALF w/ rob, LGW)', '(September 12, 2016) (NL, RLS, JS w/ MALF, rob, LGW)', '(September 10, 2016) (NL, RLS, MALF w/ rob, Baer)', '(September 8, 2016) (NL, RLS, MALF w/ alpacapatrol, LGW, Sin)', 'Solo (September 7, 2016) (NL w/ Sin, MALF, LGW, Baer, Dan, rob)', '(August 31, 2016) (NL, RLS w/ rob, LGW, Dan)', '(August 29, 2016) Nick view (NL, RLS, JS w/ rob, LGW)', '(August 25, 2016) (NL, RLS w/ rob, MALF, LGW)', '(August 24, 2016) (NL, MALF w/ rob, LGW, dan)', '(August 22, 2016) (NL, MALF w/ rob, LGW)', '(August 18, 2016) (NL, RLS w/ JS, MALF, rob, LGW)', '(August 17, 2016) (NL, RLS, CS w/ rob, Baer, LGW, Dan)', '(August 15, 2016) part 1 part 2 Nick view (NL, RLS, JS w/ rob, Baer, LGW)', '(August 10, 2016) (NL, RLS w/ rob, LGW)', '(August 8, 2016) (NL, RLS, JS w/ LGW, Baer)', '(August 4, 2016) Nick view (NL, RLS, MALF w/ rob, LGW, Sin, Baer)', '(August 3, 2016) part 1 part 2 (NL, RLS, CS w/ rob, LGW)', '(August 1, 2016) (NL, RLS, JS w/ Baer, LGW)', 'Bootleg (July 28, 2016) (NL, MALF w/ JS, LGW)', 'Solo (July 27, 2016) (NL, MALF)', 'Solo (July 25, 2016) (NL)', '(July 21, 2016) (NL, RLS w/ LGW, Sin, Mathas)', '(July 20, 2016) (NL, RLS, CS w/ Baer, LGW)', '(July 18, 2016) Nick view (NL, RLS w/ MALF, LGW)', '(July 14, 2016) (NL, RLS w/ LGW, Sin)', '(July 13, 2016) (NL, RLS, CS w/ Sin)', '(July 11, 2016) (NL, RLS w/ Baer, rob, LGW)', 'Solo (July 7, 2016) (NL w/ rob)', '(July 6, 2016) (NL, RLS, CS w/ LGW)', '(July 4, 2016) (NL, RLS, JS w/ rob, LGW)', '(June 30, 2016) (NL, RLS w/ MALF, rob, Baer, LGW)', '(June 29, 2016) (NL, RLS, CS w/ rob, LGW)', '(June 27, 2016) (NL, RLS, JS w/ rob, LGW)', '(June 23, 2016) (NL, RLS w/ rob, MALF, Dan)', '(June 20, 2016) (NL, RLS, JS w/ rob, Mathas)', '(June 9, 2016) Nick view (NL, RLS, rob w/ LGW)', '(June 8, 2016) (NL, RLS, CS w/ rob, LGW)', '(June 6, 2016) (NL, RLS, JS w/ LGW, MALF, Dan)', '(June 2, 2016) (NL, RLS w/ rob, LGW)', '(June 1, 2016) (NL, RLS, CS w/ Baer, LGW)', '(May 30, 2016) (NL, RLS, JS w/ rob, Baer, LGW)', '(May 26, 2016) (NL, RLS w/ rob, LGW, Baer)', '(May 25, 2016) (NL, RLS, CS w/ rob, MALF)', '(May 23, 2016) (NL, JS w/ rob, Dan, Sin, LGW)', 'Bootleg Solo (May 19, 2016) (NL w/ Mathas, rob, LGW, Sin)', 'Bootleg Solo (May 18, 2016) (NL w/ Sin, rob, LGW)', 'Bootleg (May 16, 2016) (NL, RLS, JS w/ rob, LGW)', '(May 12, 2016) Nick view (NL, RLS w/ rob, LGW)', '(May 11, 2016) (NL, RLS, CS w/ rob)', 'Solo (May 9, 2016) (NL w/ Sin, rob, LGW)', '(May 5, 2016) (NL, RLS w/ JS, rob)', '(May 4, 2016) (NL, RLS, rob w/ MALF)', '(May 2, 2016)(NL, RLS, JS w/ MALF)', 'Solo (April 21, 2016) (NL w/ Sin, rob, LGW)', 'Solo (April 20, 2016) (NL w/ Sin)', '(April 18, 2016) (NL, RLS, JS)', 'Bootleg (April 14, 2016) (NL, RLS w/ MALF, rob, LGW)', '(April 13, 2016) (NL, RLS w/ rob, MALF, Dan, LGW)', '(April 11, 2016) (NL, RLS, JS)', '(April 7, 2016) (NL, RLS w/ MALF, rob, LGW)', '(April 6, 2016) (NL, RLS w/ LGW)', '(April 4, 2016) (NL, RLS, JS w/ rob)', '(March 31, 2016) (NL, RLS w/ MALF, rob, LGW)', '(March 30, 2016) (NL, RLS, CS w/ JS, rob)', '(March 28, 2016) (NL, RLS w/ MALF, rob)', '(March 24, 2016) part 1 part 2 Nick view (NL, RLS w/ LGW)', '(March 23, 2016) part 1 part 2 (NL, RLS, CS w/ MALF, rob)', 'Bootleg Solo (March 3, 2016) (NL, RLS w/ rob, Dan, LGW)', '(March 2, 2016) (NL, RLS, CS w/ MALF)', '(February 29, 2016) (NL, RLS, JS w/ rob)', '[3 year NLversary!] (February 25, 2016) (NL, RLS, dan)', '(February 24, 2016) (NL, RLS, CS)', '(February 22, 2016) (NL, RLS, JS w/ rob)', '(February 18, 2016) (NL, RLS)', 'Bootleg Solo (February 17, 2016) (NL)', '(February 15, 2016) (NL, RLS, JS w/ MALF)', '(February 10, 2016) (NL, RLS, CS w/ MALF)', '(February 8, 2016) (NL, RLS, JS)', '(February 4, 2016) (NL, RLS w/ MALF)', '(February 3, 2016) (NL, RLS, CS w/ MALF)', '(February 1, 2016) (NL, RLS, JS)', '(January 28, 2016) (NL, RLS, MALF)', '(January 27, 2016) (NL, RLS)', '(January 25, 2016) (NL, RLS, JS, MALF)', '(January 21, 2016) (NL, RLS, MALF w/ rob)', '(January 20, 2016) (NL, RLS w/ rob)', '(January 18, 2016) (NL, RLS, JS)', '(January 13, 2016) (NL, RLS, CS w/ rob)', '(January 11, 2016) (NL, RLS, JS w/ rob)', '(January 7, 2016) (NL, RLS w/ MALF, rob)', '(January 6, 2016) (NL, RLS, CS w/ rob)', '(January 4, 2016) Nick view (NL, RLS, JS w/ rob)', '(December 31, 2015) (NL, RLS w/ rob)', '(December 30, 2015) (NL, RLS, CS)', '(December 28, 2015)(NL, RLS, JS)', 'Solo (December 24, 2015) (NL w/ Kate)', '(December 23, 2015) part 1 part 2 (NL, RLS, CS)', '(December 21, 2015) (NL, RLS, JS, w/ MALF)', 'Solo (December 19, 2015) (NL w/ Kate, Baer)', 'Solo (December 18, 2015) (NL w/ Kate)', '(December 10, 2015) (NL, RLS, MALF)', '(December 9, 2015) (NL, RLS, CS w/ rob)', '(December 7, 2015) part 1 part 2 (NL, RLS)', '(December 2, 2015) (NL, RLS, CS w/ rob)', '(November 30, 2015) part 1 part 2 (NL, RLS, MALF w/ rob)', 'Solo (November 26, 2015) (NL)', '(November 25, 2015) (NL, RLS, CS w/ rob)', '(November 23, 2015) (NL, RLS, JS w/ rob)', '(November 19, 2015) (NL, RLS, MALF)', '(November 18, 2015) part 1 part 2 (NL, RLS w/ Baer, rob)', '(November 16, 2015) (NL, RLS, JS)', '(November 12, 2015) (NL, RLS w/ rob, Baer)', '(November 11, 2015) (NL, RLS, CS w/ rob)', '(November 9, 2015) (NL, RLS, JS w/ Baer)', '(November 5, 2015) (NL, RLS, MALF)', '(November 4, 2015) (NL, RLS, CS)', '(November 2, 2015) part 1 part 2 (NL, RLS, JS)', '(October 29, 2015) (NL, RLS)', 'Bootleg Solo (October 28, 2015) part 1, part 2, part 3 (NL)', 'Bootleg (October 26, 2015) part 1 part 2 part 3 (NL, JS, MALF)', 'Bootleg Solo (October 22, 2015) part 1, part 2, part 3 (NL)', 'Solo (October 21, 2015) part 1 part 2 (NL)', '(October 19, 2015) part 1 part 2 (NL, JS, MALF)', '(October 15, 2015) part 1 part 2 (NL, MALF)', '(October 14, 2015) part 1 part 2 (NL, MALF)', 'October 7, 2015 (NL, RLS, CS w/ rob)', '(October 5, 2015) part 1 part 2 (NL, RLS, JS w/ rob)', '(October 1, 2015) part 1 part 2 (NL, RLS w/ rob, Dan)', '(September 30, 2015) part 1 part 2 (NL, RLS, CS w/ rob)', '(September 28, 2015) part 1 part 2 (NL, RLS, JS)', '(September 24, 2015) (NL, RLS)', '(September 23, 2015) (NL, RLS, CS w/ rob)', '(September 21, 2015) (NL, RLS, JS w/ rob)', '(September 17, 2015) part 1 part 2 Nick view (NL, RLS)', '(September 16, 2015) part 1 part 2 (NL, RLS, CS)', '(September 14, 2015) part 1 part 2 (NL, RLS, JS)', '(September 10, 2015) (NL, RLS)', '(September 9, 2015) part 1 part 2 (NL, RLS, CS w/ rob)', '(September 7, 2015) part 1 part 2 (NL, RLS, JS)', '(September 3, 2015) part 1 part 2 (NL, RLS w/ Baer, MALF)', '(September 2, 2015) part 1 part 2 Nick view (NL, RLS, CS w/ rob)', '(August 24, 2015) (NL, RLS, JS)', '(August 20, 2015) part 1 part 2 (NL, RLS)', '(August 19, 2015) part 1 part 2 Nick view (NL, RLS, CS w/ rob)', '(August 17, 2015) part 1 part 2 Nick view (NL, RLS, JS)', '(August 13, 2015) part 1 part 2 Nick view (NL, RLS, rob)', '(August 12, 2015) part 1 part 2 Nick view (NL, RLS, CS w/ JS)', '(August 10, 2015) part 1 part 2 part 3 Nick view (NL, RLS, JS)', 'Bootleg (August 6, 2015) part 1, part 2 (NL, RLS w/ rob, Baer, MALF)', '(August 5, 2015) part 1 part 2 Nick view (NL, RLS, CS)', '(August 3, 2015) part 1 part 2 (NL, RLS, JS w/ rob)', '(July 30, 2015) part 1 part 2 Nick view (NL, RLS)', '(July 29, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Mathas)', '(July 27, 2015) part 1 part 2 (NL, RLS, JS)', '(July 16, 2015) part 1 part 2 Nick view (NL, RLS w/ Brex)', '(July 15, 2015) part 1 part 2 Nick view (NL, RLS, CS)', '(July 13, 2015) part 1 part 2 Nick view (NL, RLS, JS w/ Baer)', '(July 9, 2015) part 1 part 2 part 3 Nick view (NL, RLS w/ Baer, MALF)', '(July 8, 2015) Part 1 Part 2 (NL, RLS, CS)', '(July 6, 2015) part 1 part 2 (NL, RLS, JS)', '(July 2, 2015) part 1 part 2 Nick view (NL, RLS w/ rob)', '(July 1, 2015) Nick view (NL, RLS, CS w/ rob)', '(June 29, 2015) part 1 part 2 Nick view (NL, RLS, JS)', '(June 25, 2015) Nick view (NL, RLS w/ rob)', '(June 24, 2015) Nick view (NL, RLS, CS w/ rob)', '(June 22, 2015) part 1 part 2 Nick view (NL, RLS, JS)', '(June 18, 2015) part 1 part 2 (NL, RLS)', '(June 17, 2015) part 1 part 2 (NL, RLS)', '(June 15, 2015) part 1 part 2 (NL, RLS, JS w/ MALF)', '(June 11, 2015) part 1 part 2 (NL, RLS w/ rob, MALF)', '(June 10, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(June 8, 2015) part 1 part 2 (NL, RLS w/ rob)', '(May 28, 2015) part 1 part 2 (NL w/ Baer, Fox)', '(May 27, 2015) part 1 part 2 (NL)', '(May 25, 2015) part 1 part 2 (NL, Arumba)', '(May 21, 2015) part 1 part 2 (NL, RLS)', '(May 20, 2015) part 1 part 2 (NL, RLS)', 'Bootleg (May 18, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', 'Bootleg (May 14, 2015) part 1 part 2 part 3 (NL, RLS)', 'Bootleg (May 13, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', 'Bootleg (May 11, 2015) part 1 part 2 part 3 (NL, RLS)', '(May 7, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(May 6, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(May 4, 2015) Part 1 part 2 (NL, RLS w/ rob, Baer)', 'Bootleg Solo (April 23, 2015) part 2 part 3 (NL)', 'Bootleg (April 22, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', '(April 20, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', 'Bootleg (April 16, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', '(April 15, 2015) part 1 part 2 (NL, RLS w/ cobaltstreak, baer, rob)', 'Bootleg (April 13, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, Baer)', '(April 9, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(April 8, 2015) part 1 part 2 part 3 (NL, RLS w/ rob)', '(April 6, 2015) part 1 part 2 (NL, RLS)', '(April 2, 2015) part 1 part 2 (NL, RLS w/ Baer)', '(April 1, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(March 26, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', 'Bootleg (March 25, 2015) part 1 part 2 part 3 (NL, RLS)', '(March 23, 2015) part 1 part 2 (NL, RLS)', '(March 19, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', '(March 18, 2015) part 1 part 2 (NL, RLS)', 'Solo (March 12, 2015) part 1 part 2 (NL)', '(March 11, 2015) part 1 part 2 (NL, RLS w/ rob, Baer)', 'Bootleg (February 26, 2015) part 1 part 2 part 3 (NL, RLS w/ rob, fox)', '[2 year NLversary!] Bootleg (February 25, 2015) part 1 part 2 part 3 (NL, RLS w/ JS, rob)', '(February 23, 2015) part 1 part 2 (NL, RLS w/ JS, rob)', 'Bootleg (February 19, 2015) part 1 part 2 part 3 (NL, RLS w/ rob)', '(February 18, 2015) part 1 part 2 (NL, RLS w/ JS, rob)', '(February 16, 2015) part 1 part 2 (NL, RLS w/ JS, rob)', '(February 12, 2015) part 1 part 2 (NL, RLS)', '(February 11, 2015) part 1 part 2 (NL w/ Baer)', '(February 9, 2015) part 1 part 2 (NL, RLS)', '(February 5, 2015) part 1 part 2 part 3 (NL, RLS w/ JS, rob)', '(February 2, 2015) part 1 part 2 (NL, RLS w/ rob)', '(January 29, 2015) part 1 part 2 (NL, RLS w/ rob)', '(January 28, 2015) part 1 part 2 (NL, RLS)', 'Bootleg (January 8, 2015) part 1, part 2, part 3 (NL, RLS)', '(January 7, 2015) part 1 part 2 (NL, RLS w/ JS, rob)', '(January 5, 2015) part 1 part 2 (NL, RLS w/ rob)', '(January 1, 2015) part 1 part 2 (NL, RLS)', '(December 31, 2014) part 1 part 2 (NL, RLS)', '(December 29, 2014) part 1, part 2 (NL, RLS w/ fox)', 'Bootleg (December 22, 2014) part 1, part 2 (NL, RLS)', '(December 18, 2014) part 1, part 2 (NL)', '(December 15, 2014) part 1, part 2 (NL, RLS)', 'Bootleg (December 11, 2014) part 1 part 2 (NL, RLS w/ Kate, Baer)', '(December 10, 2014) part 1, part 2 (NL, RLS)', '(December 8, 2014) part 1, part 2 (cat cam!) (NL, RLS w/ rob, Mag)', '(December 4,2014) part 1, part 2 (NL, RLS)', '(December 3, 2014) part 1, part 2 (NL, RLS)', '(November 27, 2014) part 1, part 2 (NL)', '(November 26, 2014) part 1, part 2 (NL)', '(November 24, 2014) part 1, part 2 (NL, RLS)', '(November 20, 2014) part 1, part 2 (NL, RLS)', 'Bootleg (November 19, 2014) part 1, part 2 (NLS, RLS, JS!)', '(November 17, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(November 13, 2014) part 1, part 2 (NL, RLS)', "(November 12, 2014) part 1 Bootleg Nick's view part 2 (NL, RLS w/ rob, Baer)", 'Bootleg (November 6, 2014) part 1 part 2 (NL, RLS w/ rob, Baer)', '(November 5, 2014) part 1, part 2 (NL, RLS)', '(November 3, 2014) part 1, part 2 (NL, RLS w/ rob)', '(October 30, 2014) part 1, part 2 (NL, RLS)', '(October 29, 2014) part 1, part 2 (NL, RLS)', '(October 27, 2014) part 1, part 2 (NL, RLS w/rob, Baer)', 'Bootleg (October 23, 2014) part 1, part 2 (NL, RLS)', '(October 22, 2014) Part 1, Part 2 (NL, RLS w/ rob, MALF)', '(October 20, 2014) Part 1, Part 2 (NL, RLS w/ rob, Baer)', '(October 16, 2014) part 1, part 2 (NL, RLS)', '(October 15, 2014) part 1, part 2, part 3 (NL, RLS w/ rob, Baer)', '(October 13, 2014) part 1, part 2 (NL, RLS)', '(October 9, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(October 8, 2014) part 1, part 2 (NL, RLS)', '(October 6, 2014) part 1, part 2, part 3 (NL, RLS w/ rob)', '(October 2, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(October 1, 2014) part 1, part 2 (NL, RLS)', '(September 29, 2014) part 1, part 2 (NL)', '(September 25, 2014) part 1, part 2, part 3 (NL, RLS w/ rob, Mag)', '(September 24, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(September 22, 2014) part 1, part 2 (NL, RLS w/ rob, Mag)', '(September 18, 2014) part 1 part 2 (NL, RLS)', '(September 17, 2014) part 1 part 2 (NL, RLS w/ JS, rob)', '(September 15, 2014) part 1, part 2 (NL, RLS w/ JS, rob)', '(September 11, 2014) part 1, part 2 (NL, RLS)', '(September 10, 2014) part 1, part 2 (NL, RLS w/ JS, Mag)', '(September 8, 2014) part 1, Part 2 (NL, RLS w/ JS)', '(August 27, 2014) part 1, part 2 (NL, RLS in person)', '(August 25, 2014) part 1, part 2 (NL, RLS, Kate in person)', '(August 13, 2014) part 1, part 2 (NL, RLS)', '(August 12, 2014) part 1, part 2 (NL, RLS w/ rob)', '(August 7, 2014) part 1, part 2 (NL, RLS w/ rob)', '(August 6, 2014) part 1, part 2 (NL, RLS)', '(August 4, 2014) part 1, part 2 (NL, RLS w/ rob)', '(July 31, 2014) part 1, part 2 (NL, RLS)', '(July 30, 2014) part 1, part 2 (NL, RLS)', '(July 28, 2014) part 1, part 2 (NL, RLS w/ Kate, Baer, rob)', '(July 24, 2014) part 1, part 2 (NL, RLS w/ rob, Baer)', '(July 21, 2014) part 1, part 2 (NL, RLS)', '(July 16, 2014) part 1, part 2, part 3 (NL, RLS)', '(July 14, 2014) part 1, part 2 (NL, RLS)', '(July 10, 2014) part 1, part 2 (NL, RLS w/ Baer)', '(July 9, 2014) part 1, part 2 (NL, RLS)', '(July 7, 2014) part 1, part 2 (NL, RLS w/ Baer)', '(July 2, 2014) part 1, part 2 (NL, RLS)', '(June 30, 2014) part 1, part 2 (NL, RLS w/ Kate, rob)', '(June 26, 2014) part 1, part 2 (NL, RLS)', '(June 25, 2014) part 1, part 2 (NL, RLS)', '(June 19, 2014) part 1, part 2 (NL, RLS w/ rob)', '(June 18, 2014) part 1, part 2 (NL, RLS w/ Kate, rob, Baer, Mathas)', '(June 16, 2014) part 1, part 2 (NL, RLS, JS)', '(June 12, 2014) part 1, part 2 (NL, RLS, JS)', '(June 11, 2014) part 1, part 2 (NL, RLS, JS w/ Mathas)', '(June 9, 2014) part 1, part 2 (NL, RLS, JS)', '(June 5, 2014) part 1, part 2 (NL, RLS, JS)', '(June 4, 2014) part 1, part 2 (NL, RLS, JS)', '(June 2, 2014) part 1, part 2 (NL, RLS, JS)', '(May 29, 2014) part 1, part 2 (NL, RLS, JS)', '(May 28, 2014) part 1, part 2 (NL, RLS, JS)', '(May 26, 2014) part 1, part 2 (NL, RLS, JS)', '(May 15, 2014) part 1, part 2 (NL, RLS)', '(May 14, 2014) part 1, part 2 (NL, RLS)', '(May 12, 2014) part 1, part 2 (NL, RLS)', '(May 8, 2014) part 1, part 2 (NL, RLS, JS)', '(May 5, 2014) part 1, part 2 (NL, RLS, JS w/ Mike Bithell)', '(May 1, 2014) part 1, part 2 (NL, RLS, JS)', '(April 30, 2014) part 1, part 2 (NL, RLS, JS)', '(April 28, 2014) part 1, part 2 (NL, RLS, JS)', '(April 24, 2014) part 1, part 2 (NL, RLS, JS)', '(April 23, 2014) part 1, part 2 (NL, RLS, JS)', '(April 21, 2014) part 1, part 2 (NL, RLS, JS)', '(April 17, 2014) part 1, part 2 (NL, RLS, JS)', '(April 16, 2014) part 1, part 2 (NL, RLS, JS)', '(April 7, 2014) part 1, part 2 (NL, RLS, JS)', '(April 3, 2014) part 1, part 2 (NL, RLS, JS)', '(April 2, 2014) part 1, part 2 (NL, RLS, JS)', '(March 31, 2014) part 1, part 2 (NL, RLS, JS)', '(March 27, 2014) part 1, part 2 (NL, RLS, JS)', '(March 26, 2014) part 1, part 2 (NL, RLS, JS)', '(March 24, 2014) part 1, part 2 (NL, RLS, JS)', '(March 13, 2014) part 1, part 2 (NL, RLS, JS)', '(March 12, 2014) part 1, part 2 (NL, RLS, JS)', '(March 10, 2014) part 1, part 2 (NL, RLS, JS)', '(March 6, 2014) part 1, part 2 (NL, RLS, JS)', '(March 5, 2014) part 1, part 2 (NL, RLS, JS)', '(March 3, 2014) part 1, part 2 (NL, RLS, JS)', '(February 27, 2014) part 1, part 2 (NL, RLS, JS)', '(February 26, 2014) part 1, part 2 (NL, RLS, JS)', '(February 24, 2014) part 1, part 2 (NL, RLS, JS)', '(February 20, 2014) part 1, part 2 (NL, RLS)', '(February 19, 2014) part 1, part 2 (NL, RLS, JS)', '(February 17, 2014) part 1, part 2 (NL, RLS, JS)', '(February 13, 2014) part 1, part 2 (NL, RLS)', '(February 12, 2014) part 1, part 2, part 3, part 4, part 5 (NL, RLS)', '(February 10, 2014) part 1, part 2 (NL, RLS, JS)', '(February 6, 2014) part 1, part 2 (NL, RLS, JS)', '(February 5, 2014) part 1, part 2 (NL, RLS, JS, MALF)', '(February 3, 2014) part 1, part 2 (NL, RLS w/ rob, MALF)', '(January 30, 2014) part 1, part 2 (NL, RLS, JS w/ Mike Bithell)', '(January 29, 2014) part 1, part 2 (NL, RLS, JS)', '(January 27, 2014) (NL, RLS, JS w/ Crendor)', '(January 20, 2014) part 1, part 2 (NL, RLS, JS)', '(January 16, 2014) part 1, part 2 (NL, RLS, JS)', '(January 15, 2014) part 1, part 2 (NL, RLS, JS)', '(January 13, 2014) part 1, part 2 (NL, RLS, MALF)', '(January 9, 2014) part 1, part 2 (NL, RLS, MALF w/ rob)', '(January 8, 2014) part 1, part 2 (NL, RLS, JS)', '(January 6, 2014) part 1, part 2 (NL, RLS, JS)', '(December 19, 2013), part 1, part 2 (NL, RLS, JS)', '(December 18, 2013), part 1, part 2 (NL, JS)', '(December 16, 2013), part 1, part 2 (NL, RLS, JS)', '(December 12, 2013), part 1, part 2 (NL, RLS, JS)', '(December 11, 2013), part 1, part 2 (NL, RLS, JS)', '(December 9, 2013), part 1, part 2 (NL, RLS, JS)', '(December 5, 2013), part 1, part 2 (NL, RLS, JS)', '(December 4, 2013), part 1, part 2 (NL, RLS, JS)', '(December 2, 2013), part 1, part 2 (NL, RLS, JS)', '(November 28, 2013), part 1, part 2 (NL, JS)', '(November 27, 2013), part 1, part 2 (NL, RLS, JS)', '(November 25, 2013), part 1, Part 2 (NL, RLS, JS)', '(November 21, 2013), part 1, part 2 (NL, RLS, JS)', '(November 20, 2013), part 1, part 2 (NL, RLS, JS)', '(November 18, 2013), part 1, part 2 (NL, RLS, JS)', '(November 14, 2013), part 1, part 2 (NL, RLS, JS)', '(November 13, 2013), part 1, part 2 (NL, RLS, MALF)', '(November 11, 2013), part 1, part 2 (NL, RLS, MALF)', '(November 7, 2013), part 1, part 2 (NL, RLS, JS, MALF)', '(November 6, 2013), part 1, part 2 (NL, RLS, JS)', '(November 4, 2013), part 1, part 2 (NL, RLS, MALF)', '(October 31, 2013) part 1 part 2 (NL, RLS, JS)', '(October 30, 2013), part 1, part 2 (NL, RLS, JS)', '(October 28, 2013) (NL, RLS, JS)', '(October 24, 2013) Part 1, Part 2 (NL, RLS, JS)', '(October 23, 2013) (NL, RLS, JS w/ rob)', '(October 21, 2013) (NL, RLS, JS)', '(October 17, 2013) (NL, RLS, JS w/ rob)', '(October 16, 2013) (NL, RLS, JS w/ rob)', '(October 14, 2013) (NL, RLS, JS)', '(October 10, 2013) (NL, RLS, JS w/ RPG)', '(October 9, 2013) (NL, RLS, JS)', '(October 7, 2013) (NL, RLS, JS)', '(October 3, 2013) (NL, RLS, JS)', '(October 2, 2013) (NL, RLS, JS)', '(September 30, 2013) Part 1, Part 2 (NL, RLS, JS)', '(September 26, 2013) (NL, RLS, JS)', '(September 25, 2013) Part 1, Part 2 (NL, RLS, JS)', '(September 23, 2013) (NL, RLS, JS)', '(September 19, 2013) (NL, RLS, JS)', '(September 18, 2013) (NL, RLS, JS, MALF)', '(September 16, 2013) (NL, RLS, JS)', '(September 12, 2013) (NL, RLS, JS)', '(September 11, 2013) (NL, RLS, JS)', '(September 9, 2013) (NL, RLS, JS)', '(September 5, 2013) (NL, RLS, JS)', '(September 4, 2013) (NL, RLS, JS w/ Ohm)', '(August 26, 2013) part 1, part 2 (NL, RLS, JS)', '(August 22, 2013) (NL, RLS, JS w/ Ohm)', '(August 21, 2013) (NL, RLS, JS w/ Ohm)', '(August 19, 2013) (NL, RLS, JS)', '(August 15, 2013) (NL, RLS, JS)', '(August 14, 2013) (NL, RLS, JS)', '(August 12, 2013) (NL, RLS, JS)', '(August 1, 2013) (NL, RLS, JS w/ Kate)', '(July 31, 2013) (NL, RLS, JS w/ Kate)', '(July 29, 2013) (NL, RLS, JS w/ Ohm)', '(July 25, 2013) (NL, RLS, JS w/ Kate)', '(July 24, 2013) (NL, RLS, JS w/ Kate, MALF)', '(July 22, 2013) (NL, RLS, JS w/ Mike Bithell)', '(July 18, 2013) (NL, RLS, JS w/ Kate)', '(July 17, 2013) part 1, part 2 (NL, RLS, JS w/ Kate)', '(July 15, 2013) (NL, RLS, JS)', '(July 11, 2013) (NL, RLS, JS)', '(July 8, 2013) (NL, RLS, JS)', '(July 4, 2013) (NL, RLS, JS w/ Ohm)', '(July 3, 2013) (NL, RLS, JS w/ Kate, Ohm, rob)', '(July 1, 2013) (NL, RLS, JS)', '(June 20, 2013) (NL, RLS, JS w/ Ohm, rob, Pixel)', '(June 19, 2013) (NL, RLS, JS w/ Ohm, rob)', '(June 17, 2013) (NL, RLS, JS w/ Ohm, Green)', '(June 13, 2013) (NL, RLS, JS w/ Ohm, rob, LGW)', '(June 12, 2013) (NL, RLS, JS w/ Ohm, rob, RPG, Mathas)', '(June 10, 2013) (NL, RLS, JS w/ Ohm, rob)', '(June 5, 2013) (NL, RLS, JS w/ Ohm)', '(June 3, 2013) (NL, RLS, JS w/ Green, rob)', '(May 30, 2013) (NL, RLS, JS w/ Ohm, rob, Green)', '(May 29, 2013) (NL, RLS, JS)', '(May 27, 2013) (NL, RLS, JS w/ Ohm, rob)', '(May 23, 2013) (NL, RLS, JS w/ Kate, Ohm)', '(May 22, 2013) (NL, RLS, JS w/ Kate, Ohm, rob, Green)', '(May 20, 2013) (NL, RLS, JS w/ Kate, Ohm, Green)', '(May 16, 2013) (NL, RLS w/ Ohm, rob, Pixel)', '(May 15, 2013) (NL, RLS, Ohm)', '(May 13, 2013) (NL, RLS w/ Ohm, Mathas, rob)', '(May 2, 2013) (NL, RLS, JS w/ Ohm, RPG, Green)', '(May 1, 2013) (NL, RLS, JS w/ Ohm)', '(April 29, 2013) (NL, RLS, JS w/ Ohm, rob, Mathas, Green)', '(April 25, 2013) (NL, RLS, JS w/ rob, Green)', '(April 24, 2013) (NL, RLS, JS w/ Ohm)', '(April 22, 2013) (NL, RLS, JS w/ Ohm, rob, Green, RPG)', '(April 18, 2013) (NL, RLS w/ RPG, Green, Ohm, rob)', '(April 17, 2013) (NL, RLS, JS w/ Green, Ohm)', '(April 15, 2013) (NL, RLS, JS w/ Ohm, Green, rob, Mathas, MALF)', '(April 11, 2013) (NL, RLS, JS w/ Green, rob)', '(April 10, 2013) (NL, RLS, JS w/ Green, Ohm)', '(April 8, 2013) (NL, RLS, JS w/ Ohm, RPG, MALF)', '(April 4, 2013) (NL, RLS, JS w/ Green, Ohm)', '(April 3, 2013) (NL, RLS, JS w/ MALF, Ohm)', '(April 1, 2013) (NL, RLS, JS w/ Ohm)', '(March 28, 2013) (NL, RLS, JS w/ RPG, Ohm)', '(March 27, 2013) (NL, RLS, JS w/ Ohm)', '(March 18, 2013) (NL, RLS, JS)', '(March 14, 2013) (NL, RLS, JS w/ MALF)', '(March 13, 2013) (NL, RLS, JS w/ Ohm)', '(March 11, 2013) (NL, RLS, JS w/ Ohm)', '(March 6, 2013) (NL, RLS, JS)', '(March 4, 2013) (NL, RLS, JS)', '(February 28, 2013) (NL, RLS, JS)', '(February 27, 2013) (NL, Kate)', '(February 25, 2013) (NL)']
    


```python
#Split into list for crew and list for dates
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




```python
#Put these data frames together
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




```python
#Clean up members list
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




```python
#A list of all unique members
names = []
for entry in fullstripped:
    for user in entry:
        if user not in names:
            names.append(user)
print(names)
```

    ['NL', 'RLS', 'CS', 'rob', 'LGW', 'HCJ', 'Baer', 'JS', 'Sin', 'Dan', 'MALF', 'Kory', 'TB', 'Kate', 'Blueman', 'BaerBaer', 'Mathas', 'Crendor', 'BRex', 'GhostBill', 'alpacapatrol', 'dan', '', 'Brex', 'Fox', 'Arumba', 'baer', 'cobaltstreak', 'fox', 'Mag', 'NLS', 'JS!', 'RLS in person', 'Kate in person', 'Mike Bithell', 'RPG', 'Ohm', 'Pixel', 'Green']
    


```python
#Clean up the list so that it follows a consistent format
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




```python
nlss_df['Crew'] = guests
nlss_df.head()
nlss_df.tail()
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
      <th>583</th>
      <td>March 6, 2013</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI]</td>
      <td>[Dark Souls invasions,  Trivia,  Tomb Raider, ...</td>
    </tr>
    <tr>
      <th>584</th>
      <td>March 4, 2013</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI]</td>
      <td>[Delver's Drop with Ryan Baker and Ryan Burrel...</td>
    </tr>
    <tr>
      <th>585</th>
      <td>February 28, 2013</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI]</td>
      <td>[Dark Souls,  Trivia,  Trials Evolution,  Ask ...</td>
    </tr>
    <tr>
      <th>586</th>
      <td>February 27, 2013</td>
      <td>[Northernlion, LovelyMomo]</td>
      <td>[Dark Souls,  Trivia,  More Dark Souls,  Ask m...</td>
    </tr>
    <tr>
      <th>587</th>
      <td>February 25, 2013</td>
      <td>[Northernlion]</td>
      <td>[Runner 2,  Super House of Dead Ninjas,  Trivi...</td>
    </tr>
  </tbody>
</table>
</div>



## FULL CSV

Now import the stream stats CSV


```python
import os
import glob
print(os.getcwd())

stream_df = pd.read_csv(r'data\FullCSV\Northernlion_Full.csv',index_col=None, header=0)
stream_df.drop('GamesPlayed',axis=1, inplace=True)
stream_df[150:160]
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
      <th>RowNum</th>
      <th>StartTime</th>
      <th>EndTime</th>
      <th>ViewsGained</th>
      <th>FollowersGained</th>
      <th>MaxViewers</th>
      <th>AverageViewers</th>
      <th>FollowersPerHour</th>
      <th>ViewsPerHour</th>
      <th>LengthMinutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>150</th>
      <td>151</td>
      <td>2017-01-09 23:15:00.000</td>
      <td>2017-01-10 02:00:00.000</td>
      <td>9391</td>
      <td>119</td>
      <td>9229</td>
      <td>7986</td>
      <td>9.92</td>
      <td>782.58</td>
      <td>180</td>
    </tr>
    <tr>
      <th>151</th>
      <td>152</td>
      <td>2017-01-08 21:30:00.000</td>
      <td>2017-01-09 00:30:00.000</td>
      <td>12764</td>
      <td>129</td>
      <td>4963</td>
      <td>4648</td>
      <td>9.92</td>
      <td>981.85</td>
      <td>195</td>
    </tr>
    <tr>
      <th>152</th>
      <td>153</td>
      <td>2017-01-05 23:15:00.000</td>
      <td>2017-01-06 02:15:00.000</td>
      <td>8497</td>
      <td>117</td>
      <td>9190</td>
      <td>7994</td>
      <td>9.00</td>
      <td>653.62</td>
      <td>195</td>
    </tr>
    <tr>
      <th>153</th>
      <td>154</td>
      <td>2017-01-04 22:15:00.000</td>
      <td>2017-01-05 01:00:00.000</td>
      <td>17681</td>
      <td>172</td>
      <td>9636</td>
      <td>7779</td>
      <td>14.33</td>
      <td>1473.42</td>
      <td>180</td>
    </tr>
    <tr>
      <th>154</th>
      <td>155</td>
      <td>2017-01-04 00:30:00.000</td>
      <td>2017-01-04 04:30:00.000</td>
      <td>34987</td>
      <td>1581</td>
      <td>17249</td>
      <td>14806</td>
      <td>93.00</td>
      <td>2058.06</td>
      <td>255</td>
    </tr>
    <tr>
      <th>155</th>
      <td>156</td>
      <td>2017-01-02 23:15:00.000</td>
      <td>2017-01-03 02:15:00.000</td>
      <td>12236</td>
      <td>118</td>
      <td>8869</td>
      <td>7483</td>
      <td>9.08</td>
      <td>941.23</td>
      <td>195</td>
    </tr>
    <tr>
      <th>156</th>
      <td>157</td>
      <td>2017-01-01 21:15:00.000</td>
      <td>2017-01-02 00:30:00.000</td>
      <td>12084</td>
      <td>290</td>
      <td>7937</td>
      <td>6316</td>
      <td>20.71</td>
      <td>863.14</td>
      <td>210</td>
    </tr>
    <tr>
      <th>157</th>
      <td>158</td>
      <td>2016-12-29 23:15:00.000</td>
      <td>2016-12-30 02:15:00.000</td>
      <td>8856</td>
      <td>120</td>
      <td>7418</td>
      <td>6714</td>
      <td>9.23</td>
      <td>681.23</td>
      <td>195</td>
    </tr>
    <tr>
      <th>158</th>
      <td>159</td>
      <td>2016-12-28 22:15:00.000</td>
      <td>2016-12-29 01:00:00.000</td>
      <td>8027</td>
      <td>68</td>
      <td>7545</td>
      <td>6668</td>
      <td>5.67</td>
      <td>668.92</td>
      <td>180</td>
    </tr>
    <tr>
      <th>159</th>
      <td>160</td>
      <td>2016-12-26 23:15:00.000</td>
      <td>2016-12-27 02:15:00.000</td>
      <td>7584</td>
      <td>100</td>
      <td>7737</td>
      <td>6746</td>
      <td>7.69</td>
      <td>583.38</td>
      <td>195</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Remove extra line we don't need
stream_df.drop(154, inplace=True)
stream_df[150:160]
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
      <th>RowNum</th>
      <th>StartTime</th>
      <th>EndTime</th>
      <th>ViewsGained</th>
      <th>FollowersGained</th>
      <th>MaxViewers</th>
      <th>AverageViewers</th>
      <th>FollowersPerHour</th>
      <th>ViewsPerHour</th>
      <th>LengthMinutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>150</th>
      <td>151</td>
      <td>2017-01-09 23:15:00.000</td>
      <td>2017-01-10 02:00:00.000</td>
      <td>9391</td>
      <td>119</td>
      <td>9229</td>
      <td>7986</td>
      <td>9.92</td>
      <td>782.58</td>
      <td>180</td>
    </tr>
    <tr>
      <th>151</th>
      <td>152</td>
      <td>2017-01-08 21:30:00.000</td>
      <td>2017-01-09 00:30:00.000</td>
      <td>12764</td>
      <td>129</td>
      <td>4963</td>
      <td>4648</td>
      <td>9.92</td>
      <td>981.85</td>
      <td>195</td>
    </tr>
    <tr>
      <th>152</th>
      <td>153</td>
      <td>2017-01-05 23:15:00.000</td>
      <td>2017-01-06 02:15:00.000</td>
      <td>8497</td>
      <td>117</td>
      <td>9190</td>
      <td>7994</td>
      <td>9.00</td>
      <td>653.62</td>
      <td>195</td>
    </tr>
    <tr>
      <th>153</th>
      <td>154</td>
      <td>2017-01-04 22:15:00.000</td>
      <td>2017-01-05 01:00:00.000</td>
      <td>17681</td>
      <td>172</td>
      <td>9636</td>
      <td>7779</td>
      <td>14.33</td>
      <td>1473.42</td>
      <td>180</td>
    </tr>
    <tr>
      <th>155</th>
      <td>156</td>
      <td>2017-01-02 23:15:00.000</td>
      <td>2017-01-03 02:15:00.000</td>
      <td>12236</td>
      <td>118</td>
      <td>8869</td>
      <td>7483</td>
      <td>9.08</td>
      <td>941.23</td>
      <td>195</td>
    </tr>
    <tr>
      <th>156</th>
      <td>157</td>
      <td>2017-01-01 21:15:00.000</td>
      <td>2017-01-02 00:30:00.000</td>
      <td>12084</td>
      <td>290</td>
      <td>7937</td>
      <td>6316</td>
      <td>20.71</td>
      <td>863.14</td>
      <td>210</td>
    </tr>
    <tr>
      <th>157</th>
      <td>158</td>
      <td>2016-12-29 23:15:00.000</td>
      <td>2016-12-30 02:15:00.000</td>
      <td>8856</td>
      <td>120</td>
      <td>7418</td>
      <td>6714</td>
      <td>9.23</td>
      <td>681.23</td>
      <td>195</td>
    </tr>
    <tr>
      <th>158</th>
      <td>159</td>
      <td>2016-12-28 22:15:00.000</td>
      <td>2016-12-29 01:00:00.000</td>
      <td>8027</td>
      <td>68</td>
      <td>7545</td>
      <td>6668</td>
      <td>5.67</td>
      <td>668.92</td>
      <td>180</td>
    </tr>
    <tr>
      <th>159</th>
      <td>160</td>
      <td>2016-12-26 23:15:00.000</td>
      <td>2016-12-27 02:15:00.000</td>
      <td>7584</td>
      <td>100</td>
      <td>7737</td>
      <td>6746</td>
      <td>7.69</td>
      <td>583.38</td>
      <td>195</td>
    </tr>
    <tr>
      <th>160</th>
      <td>161</td>
      <td>2016-12-25 21:30:00.000</td>
      <td>2016-12-26 00:15:00.000</td>
      <td>7001</td>
      <td>131</td>
      <td>6298</td>
      <td>4912</td>
      <td>10.92</td>
      <td>583.42</td>
      <td>180</td>
    </tr>
  </tbody>
</table>
</div>




```python
datetime = [date.split(' ')for date in stream_df['StartTime']]
datetime
```




    [['2017-09-21', '21:15:00.000'],
     ['2017-09-20', '21:15:00.000'],
     ['2017-09-19', '20:45:00.000'],
     ['2017-09-18', '22:15:00.000'],
     ['2017-09-17', '20:30:00.000'],
     ['2017-09-14', '21:15:00.000'],
     ['2017-09-13', '21:15:00.000'],
     ['2017-09-12', '20:15:00.000'],
     ['2017-09-11', '22:00:00.000'],
     ['2017-09-10', '20:30:00.000'],
     ['2017-09-07', '21:15:00.000'],
     ['2017-09-06', '21:15:00.000'],
     ['2017-09-06', '00:45:00.000'],
     ['2017-08-30', '21:15:00.000'],
     ['2017-08-29', '20:15:00.000'],
     ['2017-08-28', '22:15:00.000'],
     ['2017-08-27', '20:15:00.000'],
     ['2017-08-24', '21:15:00.000'],
     ['2017-08-23', '21:15:00.000'],
     ['2017-08-22', '20:30:00.000'],
     ['2017-08-21', '22:15:00.000'],
     ['2017-08-20', '20:15:00.000'],
     ['2017-08-17', '21:15:00.000'],
     ['2017-08-16', '21:15:00.000'],
     ['2017-08-15', '20:15:00.000'],
     ['2017-08-14', '22:15:00.000'],
     ['2017-08-13', '20:30:00.000'],
     ['2017-08-11', '20:15:00.000'],
     ['2017-08-10', '21:15:00.000'],
     ['2017-08-07', '22:15:00.000'],
     ['2017-08-06', '20:15:00.000'],
     ['2017-08-03', '21:15:00.000'],
     ['2017-08-02', '21:15:00.000'],
     ['2017-08-01', '20:15:00.000'],
     ['2017-07-31', '22:15:00.000'],
     ['2017-07-30', '20:15:00.000'],
     ['2017-07-27', '21:15:00.000'],
     ['2017-07-26', '22:00:00.000'],
     ['2017-07-25', '22:30:00.000'],
     ['2017-07-24', '22:15:00.000'],
     ['2017-07-23', '20:45:00.000'],
     ['2017-07-22', '02:15:00.000'],
     ['2017-07-20', '21:15:00.000'],
     ['2017-07-19', '21:15:00.000'],
     ['2017-07-17', '22:15:00.000'],
     ['2017-07-16', '20:15:00.000'],
     ['2017-07-13', '21:00:00.000'],
     ['2017-07-12', '21:15:00.000'],
     ['2017-07-11', '20:00:00.000'],
     ['2017-07-10', '22:15:00.000'],
     ['2017-07-09', '20:15:00.000'],
     ['2017-07-06', '21:15:00.000'],
     ['2017-07-05', '21:15:00.000'],
     ['2017-07-04', '20:45:00.000'],
     ['2017-07-03', '22:15:00.000'],
     ['2017-06-25', '20:15:00.000'],
     ['2017-06-22', '21:15:00.000'],
     ['2017-06-21', '21:15:00.000'],
     ['2017-06-19', '22:15:00.000'],
     ['2017-06-18', '20:30:00.000'],
     ['2017-06-15', '21:15:00.000'],
     ['2017-06-14', '21:15:00.000'],
     ['2017-06-13', '21:15:00.000'],
     ['2017-06-13', '20:15:00.000'],
     ['2017-06-12', '22:15:00.000'],
     ['2017-06-11', '19:30:00.000'],
     ['2017-06-08', '21:15:00.000'],
     ['2017-06-07', '21:15:00.000'],
     ['2017-06-06', '22:15:00.000'],
     ['2017-06-05', '22:15:00.000'],
     ['2017-06-04', '18:45:00.000'],
     ['2017-06-01', '21:00:00.000'],
     ['2017-05-31', '21:15:00.000'],
     ['2017-05-30', '20:45:00.000'],
     ['2017-05-29', '22:15:00.000'],
     ['2017-05-28', '20:15:00.000'],
     ['2017-05-25', '21:15:00.000'],
     ['2017-05-24', '21:15:00.000'],
     ['2017-05-22', '22:15:00.000'],
     ['2017-05-21', '20:15:00.000'],
     ['2017-05-18', '21:15:00.000'],
     ['2017-05-17', '21:15:00.000'],
     ['2017-05-16', '21:30:00.000'],
     ['2017-05-15', '22:15:00.000'],
     ['2017-05-14', '20:15:00.000'],
     ['2017-05-11', '21:00:00.000'],
     ['2017-05-10', '21:15:00.000'],
     ['2017-05-08', '22:15:00.000'],
     ['2017-05-07', '20:00:00.000'],
     ['2017-05-04', '21:15:00.000'],
     ['2017-05-03', '21:15:00.000'],
     ['2017-05-03', '00:45:00.000'],
     ['2017-05-01', '22:15:00.000'],
     ['2017-04-30', '20:00:00.000'],
     ['2017-04-27', '21:00:00.000'],
     ['2017-04-25', '22:15:00.000'],
     ['2017-04-24', '22:15:00.000'],
     ['2017-04-23', '20:15:00.000'],
     ['2017-04-20', '22:15:00.000'],
     ['2017-04-19', '21:15:00.000'],
     ['2017-04-09', '20:30:00.000'],
     ['2017-04-06', '22:15:00.000'],
     ['2017-04-05', '21:15:00.000'],
     ['2017-04-03', '22:15:00.000'],
     ['2017-04-02', '20:15:00.000'],
     ['2017-03-30', '22:15:00.000'],
     ['2017-03-29', '21:15:00.000'],
     ['2017-03-27', '22:15:00.000'],
     ['2017-03-26', '20:15:00.000'],
     ['2017-03-23', '22:00:00.000'],
     ['2017-03-22', '21:15:00.000'],
     ['2017-03-20', '22:15:00.000'],
     ['2017-03-19', '20:30:00.000'],
     ['2017-03-16', '22:15:00.000'],
     ['2017-03-15', '21:15:00.000'],
     ['2017-03-08', '22:15:00.000'],
     ['2017-03-06', '23:15:00.000'],
     ['2017-03-05', '21:15:00.000'],
     ['2017-03-02', '23:00:00.000'],
     ['2017-03-01', '22:15:00.000'],
     ['2017-02-27', '23:15:00.000'],
     ['2017-02-26', '21:30:00.000'],
     ['2017-02-23', '23:00:00.000'],
     ['2017-02-22', '22:15:00.000'],
     ['2017-02-20', '23:15:00.000'],
     ['2017-02-19', '21:15:00.000'],
     ['2017-02-16', '23:15:00.000'],
     ['2017-02-15', '22:15:00.000'],
     ['2017-02-13', '23:15:00.000'],
     ['2017-02-12', '21:30:00.000'],
     ['2017-02-09', '23:15:00.000'],
     ['2017-02-08', '22:15:00.000'],
     ['2017-02-06', '23:15:00.000'],
     ['2017-02-05', '21:15:00.000'],
     ['2017-02-02', '23:15:00.000'],
     ['2017-02-01', '22:15:00.000'],
     ['2017-01-31', '00:15:00.000'],
     ['2017-01-29', '21:30:00.000'],
     ['2017-01-26', '23:15:00.000'],
     ['2017-01-25', '22:15:00.000'],
     ['2017-01-24', '01:30:00.000'],
     ['2017-01-23', '23:15:00.000'],
     ['2017-01-22', '21:15:00.000'],
     ['2017-01-19', '23:15:00.000'],
     ['2017-01-18', '22:15:00.000'],
     ['2017-01-16', '23:15:00.000'],
     ['2017-01-15', '22:30:00.000'],
     ['2017-01-15', '21:15:00.000'],
     ['2017-01-12', '23:15:00.000'],
     ['2017-01-11', '22:15:00.000'],
     ['2017-01-09', '23:15:00.000'],
     ['2017-01-08', '21:30:00.000'],
     ['2017-01-05', '23:15:00.000'],
     ['2017-01-04', '22:15:00.000'],
     ['2017-01-02', '23:15:00.000'],
     ['2017-01-01', '21:15:00.000'],
     ['2016-12-29', '23:15:00.000'],
     ['2016-12-28', '22:15:00.000'],
     ['2016-12-26', '23:15:00.000'],
     ['2016-12-25', '21:30:00.000'],
     ['2016-12-22', '23:15:00.000'],
     ['2016-12-21', '22:30:00.000'],
     ['2016-12-19', '23:15:00.000'],
     ['2016-12-18', '21:15:00.000'],
     ['2016-12-15', '23:15:00.000'],
     ['2016-12-14', '23:15:00.000'],
     ['2016-12-12', '23:00:00.000'],
     ['2016-12-11', '21:30:00.000'],
     ['2016-12-08', '23:15:00.000'],
     ['2016-12-07', '22:15:00.000'],
     ['2016-12-06', '23:15:00.000'],
     ['2016-12-05', '23:15:00.000'],
     ['2016-12-04', '21:15:00.000'],
     ['2016-12-02', '22:15:00.000'],
     ['2016-12-02', '21:30:00.000'],
     ['2016-12-01', '23:45:00.000'],
     ['2016-11-30', '21:15:00.000'],
     ['2016-11-28', '23:15:00.000'],
     ['2016-11-27', '21:15:00.000'],
     ['2016-11-24', '23:15:00.000'],
     ['2016-11-23', '22:15:00.000'],
     ['2016-11-21', '23:15:00.000'],
     ['2016-11-20', '21:15:00.000'],
     ['2016-11-17', '23:15:00.000'],
     ['2016-11-16', '22:15:00.000'],
     ['2016-11-14', '23:15:00.000'],
     ['2016-11-13', '21:15:00.000'],
     ['2016-11-10', '23:15:00.000'],
     ['2016-11-09', '22:15:00.000'],
     ['2016-11-07', '23:15:00.000'],
     ['2016-11-06', '21:15:00.000'],
     ['2016-11-03', '22:15:00.000'],
     ['2016-11-02', '21:45:00.000'],
     ['2016-10-31', '22:15:00.000'],
     ['2016-10-30', '20:15:00.000'],
     ['2016-10-27', '22:15:00.000'],
     ['2016-10-26', '21:15:00.000'],
     ['2016-10-24', '22:15:00.000'],
     ['2016-10-23', '20:15:00.000'],
     ['2016-10-20', '22:15:00.000'],
     ['2016-10-19', '21:15:00.000'],
     ['2016-10-18', '20:15:00.000'],
     ['2016-10-17', '22:15:00.000'],
     ['2016-10-13', '18:15:00.000'],
     ['2016-10-12', '21:15:00.000'],
     ['2016-10-10', '22:15:00.000'],
     ['2016-10-09', '20:15:00.000'],
     ['2016-10-06', '22:15:00.000'],
     ['2016-10-05', '21:15:00.000'],
     ['2016-10-03', '22:15:00.000'],
     ['2016-10-02', '20:15:00.000'],
     ['2016-09-29', '22:15:00.000'],
     ['2016-09-28', '21:15:00.000'],
     ['2016-09-26', '22:00:00.000'],
     ['2016-09-25', '20:15:00.000'],
     ['2016-09-22', '22:15:00.000'],
     ['2016-09-12', '22:15:00.000'],
     ['2016-09-12', '02:30:00.000'],
     ['2016-09-11', '20:15:00.000'],
     ['2016-09-11', '02:30:00.000'],
     ['2016-09-11', '01:15:00.000'],
     ['2016-09-08', '22:15:00.000'],
     ['2016-09-07', '21:15:00.000'],
     ['2016-08-31', '21:15:00.000'],
     ['2016-08-29', '22:15:00.000'],
     ['2016-08-28', '20:15:00.000'],
     ['2016-08-25', '22:15:00.000'],
     ['2016-08-24', '21:15:00.000'],
     ['2016-08-22', '22:15:00.000'],
     ['2016-08-21', '20:15:00.000'],
     ['2016-08-18', '22:15:00.000'],
     ['2016-08-17', '21:15:00.000'],
     ['2016-08-16', '21:15:00.000'],
     ['2016-08-15', '22:15:00.000'],
     ['2016-08-10', '21:15:00.000'],
     ['2016-08-08', '22:15:00.000'],
     ['2016-08-07', '20:15:00.000'],
     ['2016-08-04', '22:15:00.000'],
     ['2016-08-03', '21:15:00.000'],
     ['2016-08-01', '22:15:00.000'],
     ['2016-07-31', '20:15:00.000'],
     ['2016-07-29', '21:15:00.000'],
     ['2016-07-28', '22:15:00.000'],
     ['2016-07-27', '21:15:00.000'],
     ['2016-07-26', '20:15:00.000'],
     ['2016-07-25', '22:15:00.000'],
     ['2016-07-24', '20:15:00.000'],
     ['2016-07-21', '22:15:00.000'],
     ['2016-07-20', '21:15:00.000'],
     ['2016-07-18', '22:15:00.000'],
     ['2016-07-17', '20:15:00.000'],
     ['2016-07-14', '22:15:00.000'],
     ['2016-07-13', '21:15:00.000'],
     ['2016-07-11', '22:15:00.000'],
     ['2016-07-10', '20:00:00.000'],
     ['2016-07-07', '22:15:00.000'],
     ['2016-07-06', '21:15:00.000'],
     ['2016-07-04', '22:15:00.000'],
     ['2016-07-03', '20:15:00.000'],
     ['2016-06-30', '22:15:00.000'],
     ['2016-06-29', '21:15:00.000'],
     ['2016-06-27', '22:15:00.000'],
     ['2016-06-26', '20:15:00.000'],
     ['2016-06-23', '22:15:00.000'],
     ['2016-06-20', '22:15:00.000'],
     ['2016-06-19', '20:30:00.000'],
     ['2016-06-11', '20:15:00.000'],
     ['2016-06-09', '22:15:00.000'],
     ['2016-06-08', '21:15:00.000'],
     ['2016-06-06', '22:15:00.000'],
     ['2016-06-05', '20:15:00.000'],
     ['2016-06-02', '22:15:00.000'],
     ['2016-06-01', '21:15:00.000'],
     ['2016-05-30', '22:15:00.000'],
     ['2016-05-29', '20:15:00.000'],
     ['2016-05-26', '22:15:00.000'],
     ['2016-05-25', '21:15:00.000'],
     ['2016-05-23', '22:15:00.000'],
     ['2016-05-22', '20:15:00.000'],
     ['2016-05-19', '22:15:00.000'],
     ['2016-05-18', '21:15:00.000'],
     ['2016-05-16', '22:15:00.000'],
     ['2016-05-15', '20:15:00.000'],
     ['2016-05-12', '22:15:00.000'],
     ['2016-05-11', '21:15:00.000'],
     ['2016-05-09', '22:15:00.000'],
     ['2016-05-08', '20:15:00.000'],
     ['2016-05-05', '22:15:00.000'],
     ['2016-05-04', '21:15:00.000'],
     ['2016-05-03', '20:45:00.000'],
     ['2016-05-02', '22:15:00.000'],
     ['2016-04-21', '22:15:00.000'],
     ['2016-04-20', '21:45:00.000'],
     ['2016-04-20', '01:15:00.000'],
     ['2016-04-18', '22:00:00.000'],
     ['2016-04-17', '20:15:00.000'],
     ['2016-04-14', '22:15:00.000'],
     ['2016-04-13', '21:15:00.000'],
     ['2016-04-12', '20:30:00.000'],
     ['2016-04-11', '22:15:00.000'],
     ['2016-04-10', '20:15:00.000'],
     ['2016-04-07', '22:15:00.000'],
     ['2016-04-06', '21:15:00.000'],
     ['2016-04-04', '22:15:00.000'],
     ['2016-04-03', '20:15:00.000'],
     ['2016-03-31', '22:15:00.000'],
     ['2016-03-30', '21:15:00.000'],
     ['2016-03-28', '22:15:00.000'],
     ['2016-03-27', '20:15:00.000'],
     ['2016-03-24', '22:15:00.000'],
     ['2016-03-23', '21:15:00.000'],
     ['2016-03-04', '00:15:00.000'],
     ['2016-03-02', '22:15:00.000'],
     ['2016-02-29', '23:15:00.000'],
     ['2016-02-28', '21:15:00.000'],
     ['2016-02-25', '23:15:00.000'],
     ['2016-02-24', '22:15:00.000'],
     ['2016-02-22', '23:15:00.000'],
     ['2016-02-21', '21:15:00.000'],
     ['2016-02-18', '23:15:00.000'],
     ['2016-02-17', '22:15:00.000'],
     ['2016-02-15', '23:15:00.000'],
     ['2016-02-14', '21:15:00.000'],
     ['2016-02-10', '22:15:00.000'],
     ['2016-02-08', '23:15:00.000'],
     ['2016-02-07', '19:45:00.000'],
     ['2016-02-04', '23:15:00.000'],
     ['2016-02-03', '22:15:00.000'],
     ['2016-02-01', '23:15:00.000'],
     ['2016-01-31', '18:15:00.000'],
     ['2016-01-28', '23:15:00.000'],
     ['2016-01-27', '22:15:00.000'],
     ['2016-01-25', '23:15:00.000'],
     ['2016-01-24', '21:15:00.000'],
     ['2016-01-21', '23:15:00.000'],
     ['2016-01-20', '22:15:00.000'],
     ['2016-01-19', '00:00:00.000'],
     ['2016-01-13', '22:15:00.000'],
     ['2016-01-11', '23:00:00.000'],
     ['2016-01-10', '21:15:00.000'],
     ['2016-01-07', '23:15:00.000'],
     ['2016-01-06', '22:15:00.000'],
     ['2016-01-04', '23:15:00.000'],
     ['2016-01-03', '21:15:00.000'],
     ['2015-12-31', '23:15:00.000'],
     ['2015-12-30', '22:15:00.000'],
     ['2015-12-28', '23:15:00.000'],
     ['2015-12-27', '21:15:00.000'],
     ['2015-12-24', '23:15:00.000'],
     ['2015-12-23', '22:15:00.000'],
     ['2015-12-21', '23:15:00.000'],
     ['2015-12-20', '21:15:00.000'],
     ['2015-12-20', '01:15:00.000'],
     ['2015-12-18', '23:15:00.000'],
     ['2015-12-13', '21:15:00.000'],
     ['2015-12-11', '00:30:00.000'],
     ['2015-12-10', '23:15:00.000'],
     ['2015-12-09', '22:15:00.000'],
     ['2015-12-07', '23:30:00.000'],
     ['2015-12-06', '21:15:00.000'],
     ['2015-12-02', '22:15:00.000'],
     ['2015-11-30', '23:15:00.000'],
     ['2015-11-29', '21:15:00.000'],
     ['2015-11-26', '23:15:00.000'],
     ['2015-11-25', '22:15:00.000'],
     ['2015-11-23', '23:15:00.000'],
     ['2015-11-22', '21:15:00.000'],
     ['2015-11-19', '23:15:00.000'],
     ['2015-11-18', '22:15:00.000'],
     ['2015-11-16', '23:15:00.000'],
     ['2015-11-15', '21:15:00.000'],
     ['2015-11-12', '23:15:00.000'],
     ['2015-11-11', '22:15:00.000'],
     ['2015-11-09', '23:15:00.000'],
     ['2015-11-09', '18:15:00.000'],
     ['2015-11-08', '21:15:00.000'],
     ['2015-11-05', '23:15:00.000'],
     ['2015-11-04', '22:15:00.000'],
     ['2015-11-02', '23:15:00.000'],
     ['2015-11-01', '20:15:00.000'],
     ['2015-10-30', '20:30:00.000'],
     ['2015-10-29', '22:15:00.000'],
     ['2015-10-28', '21:15:00.000'],
     ['2015-10-26', '22:15:00.000'],
     ['2015-10-26', '00:45:00.000'],
     ['2015-10-22', '22:15:00.000'],
     ['2015-10-21', '21:15:00.000'],
     ['2015-10-19', '22:15:00.000'],
     ['2015-10-18', '20:15:00.000'],
     ['2015-10-15', '22:15:00.000'],
     ['2015-10-14', '21:00:00.000'],
     ['2015-10-11', '20:15:00.000'],
     ['2015-10-07', '21:15:00.000'],
     ['2015-10-05', '23:15:00.000'],
     ['2015-10-04', '21:15:00.000'],
     ['2015-10-01', '22:15:00.000'],
     ['2015-09-30', '21:15:00.000'],
     ['2015-09-29', '04:15:00.000'],
     ['2015-09-28', '22:15:00.000'],
     ['2015-09-27', '20:15:00.000'],
     ['2015-09-24', '22:15:00.000'],
     ['2015-09-23', '21:15:00.000'],
     ['2015-09-22', '04:15:00.000'],
     ['2015-09-21', '22:15:00.000'],
     ['2015-09-20', '20:15:00.000'],
     ['2015-09-17', '22:15:00.000'],
     ['2015-09-16', '21:15:00.000'],
     ['2015-09-15', '04:15:00.000'],
     ['2015-09-14', '22:15:00.000'],
     ['2015-09-13', '22:15:00.000'],
     ['2015-09-10', '22:15:00.000'],
     ['2015-09-09', '22:15:00.000'],
     ['2015-09-07', '22:15:00.000'],
     ['2015-09-06', '22:30:00.000'],
     ['2015-09-06', '19:15:00.000'],
     ['2015-09-05', '19:30:00.000'],
     ['2015-09-03', '22:15:00.000'],
     ['2015-09-02', '21:15:00.000'],
     ['2015-08-24', '22:15:00.000'],
     ['2015-08-23', '21:30:00.000'],
     ['2015-08-23', '18:15:00.000'],
     ['2015-08-20', '22:00:00.000'],
     ['2015-08-19', '21:15:00.000'],
     ['2015-08-17', '22:15:00.000'],
     ['2015-08-16', '20:15:00.000'],
     ['2015-08-13', '22:15:00.000'],
     ['2015-08-12', '21:15:00.000'],
     ['2015-08-10', '22:15:00.000'],
     ['2015-08-09', '20:15:00.000'],
     ['2015-08-06', '22:15:00.000'],
     ['2015-08-05', '21:15:00.000'],
     ['2015-08-03', '22:15:00.000'],
     ['2015-08-02', '20:15:00.000'],
     ['2015-07-30', '22:15:00.000'],
     ['2015-07-30', '00:30:00.000'],
     ['2015-07-29', '22:15:00.000'],
     ['2015-07-27', '22:15:00.000'],
     ['2015-07-26', '20:15:00.000'],
     ['2015-07-20', '01:00:00.000'],
     ['2015-07-19', '20:15:00.000']]




```python
months = {'01':'January', '02':'February', '03':'March', '04':'April', '05':'May', '06':'June', '07':'July', '08':'August', '09':'September', '10':'October', '11':'November', '12':'December'}
```


```python
#Make the dates in the CSV match the format of the docket
formatted = []
for stream in datetime:
    ymd = stream[0].split('-')
    entry = months.get(ymd[1])+" "+ymd[2]+", "+ymd[0]
    formatted.append(entry)
formatted
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
     'September 07, 2017',
     'September 06, 2017',
     'September 06, 2017',
     'August 30, 2017',
     'August 29, 2017',
     'August 28, 2017',
     'August 27, 2017',
     'August 24, 2017',
     'August 23, 2017',
     'August 22, 2017',
     'August 21, 2017',
     'August 20, 2017',
     'August 17, 2017',
     'August 16, 2017',
     'August 15, 2017',
     'August 14, 2017',
     'August 13, 2017',
     'August 11, 2017',
     'August 10, 2017',
     'August 07, 2017',
     'August 06, 2017',
     'August 03, 2017',
     'August 02, 2017',
     'August 01, 2017',
     'July 31, 2017',
     'July 30, 2017',
     'July 27, 2017',
     'July 26, 2017',
     'July 25, 2017',
     'July 24, 2017',
     'July 23, 2017',
     'July 22, 2017',
     'July 20, 2017',
     'July 19, 2017',
     'July 17, 2017',
     'July 16, 2017',
     'July 13, 2017',
     'July 12, 2017',
     'July 11, 2017',
     'July 10, 2017',
     'July 09, 2017',
     'July 06, 2017',
     'July 05, 2017',
     'July 04, 2017',
     'July 03, 2017',
     'June 25, 2017',
     'June 22, 2017',
     'June 21, 2017',
     'June 19, 2017',
     'June 18, 2017',
     'June 15, 2017',
     'June 14, 2017',
     'June 13, 2017',
     'June 13, 2017',
     'June 12, 2017',
     'June 11, 2017',
     'June 08, 2017',
     'June 07, 2017',
     'June 06, 2017',
     'June 05, 2017',
     'June 04, 2017',
     'June 01, 2017',
     'May 31, 2017',
     'May 30, 2017',
     'May 29, 2017',
     'May 28, 2017',
     'May 25, 2017',
     'May 24, 2017',
     'May 22, 2017',
     'May 21, 2017',
     'May 18, 2017',
     'May 17, 2017',
     'May 16, 2017',
     'May 15, 2017',
     'May 14, 2017',
     'May 11, 2017',
     'May 10, 2017',
     'May 08, 2017',
     'May 07, 2017',
     'May 04, 2017',
     'May 03, 2017',
     'May 03, 2017',
     'May 01, 2017',
     'April 30, 2017',
     'April 27, 2017',
     'April 25, 2017',
     'April 24, 2017',
     'April 23, 2017',
     'April 20, 2017',
     'April 19, 2017',
     'April 09, 2017',
     'April 06, 2017',
     'April 05, 2017',
     'April 03, 2017',
     'April 02, 2017',
     'March 30, 2017',
     'March 29, 2017',
     'March 27, 2017',
     'March 26, 2017',
     'March 23, 2017',
     'March 22, 2017',
     'March 20, 2017',
     'March 19, 2017',
     'March 16, 2017',
     'March 15, 2017',
     'March 08, 2017',
     'March 06, 2017',
     'March 05, 2017',
     'March 02, 2017',
     'March 01, 2017',
     'February 27, 2017',
     'February 26, 2017',
     'February 23, 2017',
     'February 22, 2017',
     'February 20, 2017',
     'February 19, 2017',
     'February 16, 2017',
     'February 15, 2017',
     'February 13, 2017',
     'February 12, 2017',
     'February 09, 2017',
     'February 08, 2017',
     'February 06, 2017',
     'February 05, 2017',
     'February 02, 2017',
     'February 01, 2017',
     'January 31, 2017',
     'January 29, 2017',
     'January 26, 2017',
     'January 25, 2017',
     'January 24, 2017',
     'January 23, 2017',
     'January 22, 2017',
     'January 19, 2017',
     'January 18, 2017',
     'January 16, 2017',
     'January 15, 2017',
     'January 15, 2017',
     'January 12, 2017',
     'January 11, 2017',
     'January 09, 2017',
     'January 08, 2017',
     'January 05, 2017',
     'January 04, 2017',
     'January 02, 2017',
     'January 01, 2017',
     'December 29, 2016',
     'December 28, 2016',
     'December 26, 2016',
     'December 25, 2016',
     'December 22, 2016',
     'December 21, 2016',
     'December 19, 2016',
     'December 18, 2016',
     'December 15, 2016',
     'December 14, 2016',
     'December 12, 2016',
     'December 11, 2016',
     'December 08, 2016',
     'December 07, 2016',
     'December 06, 2016',
     'December 05, 2016',
     'December 04, 2016',
     'December 02, 2016',
     'December 02, 2016',
     'December 01, 2016',
     'November 30, 2016',
     'November 28, 2016',
     'November 27, 2016',
     'November 24, 2016',
     'November 23, 2016',
     'November 21, 2016',
     'November 20, 2016',
     'November 17, 2016',
     'November 16, 2016',
     'November 14, 2016',
     'November 13, 2016',
     'November 10, 2016',
     'November 09, 2016',
     'November 07, 2016',
     'November 06, 2016',
     'November 03, 2016',
     'November 02, 2016',
     'October 31, 2016',
     'October 30, 2016',
     'October 27, 2016',
     'October 26, 2016',
     'October 24, 2016',
     'October 23, 2016',
     'October 20, 2016',
     'October 19, 2016',
     'October 18, 2016',
     'October 17, 2016',
     'October 13, 2016',
     'October 12, 2016',
     'October 10, 2016',
     'October 09, 2016',
     'October 06, 2016',
     'October 05, 2016',
     'October 03, 2016',
     'October 02, 2016',
     'September 29, 2016',
     'September 28, 2016',
     'September 26, 2016',
     'September 25, 2016',
     'September 22, 2016',
     'September 12, 2016',
     'September 12, 2016',
     'September 11, 2016',
     'September 11, 2016',
     'September 11, 2016',
     'September 08, 2016',
     'September 07, 2016',
     'August 31, 2016',
     'August 29, 2016',
     'August 28, 2016',
     'August 25, 2016',
     'August 24, 2016',
     'August 22, 2016',
     'August 21, 2016',
     'August 18, 2016',
     'August 17, 2016',
     'August 16, 2016',
     'August 15, 2016',
     'August 10, 2016',
     'August 08, 2016',
     'August 07, 2016',
     'August 04, 2016',
     'August 03, 2016',
     'August 01, 2016',
     'July 31, 2016',
     'July 29, 2016',
     'July 28, 2016',
     'July 27, 2016',
     'July 26, 2016',
     'July 25, 2016',
     'July 24, 2016',
     'July 21, 2016',
     'July 20, 2016',
     'July 18, 2016',
     'July 17, 2016',
     'July 14, 2016',
     'July 13, 2016',
     'July 11, 2016',
     'July 10, 2016',
     'July 07, 2016',
     'July 06, 2016',
     'July 04, 2016',
     'July 03, 2016',
     'June 30, 2016',
     'June 29, 2016',
     'June 27, 2016',
     'June 26, 2016',
     'June 23, 2016',
     'June 20, 2016',
     'June 19, 2016',
     'June 11, 2016',
     'June 09, 2016',
     'June 08, 2016',
     'June 06, 2016',
     'June 05, 2016',
     'June 02, 2016',
     'June 01, 2016',
     'May 30, 2016',
     'May 29, 2016',
     'May 26, 2016',
     'May 25, 2016',
     'May 23, 2016',
     'May 22, 2016',
     'May 19, 2016',
     'May 18, 2016',
     'May 16, 2016',
     'May 15, 2016',
     'May 12, 2016',
     'May 11, 2016',
     'May 09, 2016',
     'May 08, 2016',
     'May 05, 2016',
     'May 04, 2016',
     'May 03, 2016',
     'May 02, 2016',
     'April 21, 2016',
     'April 20, 2016',
     'April 20, 2016',
     'April 18, 2016',
     'April 17, 2016',
     'April 14, 2016',
     'April 13, 2016',
     'April 12, 2016',
     'April 11, 2016',
     'April 10, 2016',
     'April 07, 2016',
     'April 06, 2016',
     'April 04, 2016',
     'April 03, 2016',
     'March 31, 2016',
     'March 30, 2016',
     'March 28, 2016',
     'March 27, 2016',
     'March 24, 2016',
     'March 23, 2016',
     'March 04, 2016',
     'March 02, 2016',
     'February 29, 2016',
     'February 28, 2016',
     'February 25, 2016',
     'February 24, 2016',
     'February 22, 2016',
     'February 21, 2016',
     'February 18, 2016',
     'February 17, 2016',
     'February 15, 2016',
     'February 14, 2016',
     'February 10, 2016',
     'February 08, 2016',
     'February 07, 2016',
     'February 04, 2016',
     'February 03, 2016',
     'February 01, 2016',
     'January 31, 2016',
     'January 28, 2016',
     'January 27, 2016',
     'January 25, 2016',
     'January 24, 2016',
     'January 21, 2016',
     'January 20, 2016',
     'January 19, 2016',
     'January 13, 2016',
     'January 11, 2016',
     'January 10, 2016',
     'January 07, 2016',
     'January 06, 2016',
     'January 04, 2016',
     'January 03, 2016',
     'December 31, 2015',
     'December 30, 2015',
     'December 28, 2015',
     'December 27, 2015',
     'December 24, 2015',
     'December 23, 2015',
     'December 21, 2015',
     'December 20, 2015',
     'December 20, 2015',
     'December 18, 2015',
     'December 13, 2015',
     'December 11, 2015',
     'December 10, 2015',
     'December 09, 2015',
     'December 07, 2015',
     'December 06, 2015',
     'December 02, 2015',
     'November 30, 2015',
     'November 29, 2015',
     'November 26, 2015',
     'November 25, 2015',
     'November 23, 2015',
     'November 22, 2015',
     'November 19, 2015',
     'November 18, 2015',
     'November 16, 2015',
     'November 15, 2015',
     'November 12, 2015',
     'November 11, 2015',
     'November 09, 2015',
     'November 09, 2015',
     'November 08, 2015',
     'November 05, 2015',
     'November 04, 2015',
     'November 02, 2015',
     'November 01, 2015',
     'October 30, 2015',
     'October 29, 2015',
     'October 28, 2015',
     'October 26, 2015',
     'October 26, 2015',
     'October 22, 2015',
     'October 21, 2015',
     'October 19, 2015',
     'October 18, 2015',
     'October 15, 2015',
     'October 14, 2015',
     'October 11, 2015',
     'October 07, 2015',
     'October 05, 2015',
     'October 04, 2015',
     'October 01, 2015',
     'September 30, 2015',
     'September 29, 2015',
     'September 28, 2015',
     'September 27, 2015',
     'September 24, 2015',
     'September 23, 2015',
     'September 22, 2015',
     'September 21, 2015',
     'September 20, 2015',
     'September 17, 2015',
     'September 16, 2015',
     'September 15, 2015',
     'September 14, 2015',
     'September 13, 2015',
     'September 10, 2015',
     'September 09, 2015',
     'September 07, 2015',
     'September 06, 2015',
     'September 06, 2015',
     'September 05, 2015',
     'September 03, 2015',
     'September 02, 2015',
     'August 24, 2015',
     'August 23, 2015',
     'August 23, 2015',
     'August 20, 2015',
     'August 19, 2015',
     'August 17, 2015',
     'August 16, 2015',
     'August 13, 2015',
     'August 12, 2015',
     'August 10, 2015',
     'August 09, 2015',
     'August 06, 2015',
     'August 05, 2015',
     'August 03, 2015',
     'August 02, 2015',
     'July 30, 2015',
     'July 30, 2015',
     'July 29, 2015',
     'July 27, 2015',
     'July 26, 2015',
     'July 20, 2015',
     'July 19, 2015']




```python
stream_df['Date'] = formatted
stream_df.head()
len(stream_df)
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
      <th>RowNum</th>
      <th>StartTime</th>
      <th>EndTime</th>
      <th>ViewsGained</th>
      <th>FollowersGained</th>
      <th>MaxViewers</th>
      <th>AverageViewers</th>
      <th>FollowersPerHour</th>
      <th>ViewsPerHour</th>
      <th>LengthMinutes</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2017-09-21 21:15:00.000</td>
      <td>2017-09-22 00:15:00.000</td>
      <td>5984</td>
      <td>36</td>
      <td>5390</td>
      <td>4636</td>
      <td>2.77</td>
      <td>460.31</td>
      <td>195</td>
      <td>September 21, 2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2017-09-20 21:15:00.000</td>
      <td>2017-09-21 00:00:00.000</td>
      <td>4881</td>
      <td>49</td>
      <td>5988</td>
      <td>5245</td>
      <td>4.08</td>
      <td>406.75</td>
      <td>180</td>
      <td>September 20, 2017</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2017-09-19 20:45:00.000</td>
      <td>2017-09-19 23:15:00.000</td>
      <td>2160</td>
      <td>48</td>
      <td>3319</td>
      <td>2718</td>
      <td>4.36</td>
      <td>196.36</td>
      <td>165</td>
      <td>September 19, 2017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2017-09-18 22:15:00.000</td>
      <td>2017-09-19 01:15:00.000</td>
      <td>4793</td>
      <td>45</td>
      <td>5325</td>
      <td>4911</td>
      <td>3.46</td>
      <td>368.69</td>
      <td>195</td>
      <td>September 18, 2017</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2017-09-17 20:30:00.000</td>
      <td>2017-09-17 23:30:00.000</td>
      <td>3380</td>
      <td>97</td>
      <td>3527</td>
      <td>3074</td>
      <td>7.46</td>
      <td>260.00</td>
      <td>195</td>
      <td>September 17, 2017</td>
    </tr>
  </tbody>
</table>
</div>






    440




```python
combined = pd.merge(nlss_df, stream_df)
combined.head()
combined.tail()
len(combined)
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
      <th>RowNum</th>
      <th>StartTime</th>
      <th>EndTime</th>
      <th>ViewsGained</th>
      <th>FollowersGained</th>
      <th>MaxViewers</th>
      <th>AverageViewers</th>
      <th>FollowersPerHour</th>
      <th>ViewsPerHour</th>
      <th>LengthMinutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>August 24, 2017</td>
      <td>[Northernlion, RockLeeSmile, CobaltStreak, Alp...</td>
      <td>[Passpartout,  Party Panic,  Pinturillo]</td>
      <td>18</td>
      <td>2017-08-24 21:15:00.000</td>
      <td>2017-08-25 00:15:00.000</td>
      <td>5400</td>
      <td>91</td>
      <td>5751</td>
      <td>4715</td>
      <td>7.00</td>
      <td>415.38</td>
      <td>195</td>
    </tr>
    <tr>
      <th>1</th>
      <td>August 23, 2017</td>
      <td>[Northernlion, RockLeeSmile, LastGreyWolf, HCJ...</td>
      <td>[Absolver,  Golf It,  Quiplash]</td>
      <td>19</td>
      <td>2017-08-23 21:15:00.000</td>
      <td>2017-08-24 00:15:00.000</td>
      <td>5962</td>
      <td>170</td>
      <td>6185</td>
      <td>5214</td>
      <td>13.08</td>
      <td>458.62</td>
      <td>195</td>
    </tr>
    <tr>
      <th>2</th>
      <td>August 21, 2017</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI, Alpaca...</td>
      <td>[Fire Pro Wrestling World,  Ultimate Chicken H...</td>
      <td>21</td>
      <td>2017-08-21 22:15:00.000</td>
      <td>2017-08-22 01:15:00.000</td>
      <td>5065</td>
      <td>95</td>
      <td>5347</td>
      <td>4797</td>
      <td>7.31</td>
      <td>389.62</td>
      <td>195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>August 17, 2017</td>
      <td>[RockLeeSmile, LastGreyWolf, HCJustin, BaerTaf...</td>
      <td>[Geoguessr,  Golf It,  Quiplash]</td>
      <td>23</td>
      <td>2017-08-17 21:15:00.000</td>
      <td>2017-08-18 00:15:00.000</td>
      <td>5365</td>
      <td>108</td>
      <td>6075</td>
      <td>5214</td>
      <td>8.31</td>
      <td>412.69</td>
      <td>195</td>
    </tr>
    <tr>
      <th>4</th>
      <td>August 16, 2017</td>
      <td>[Northernlion, BaerTaffy, LastGreyWolf, DanGhe...</td>
      <td>[Nidhogg 2,  Speedrunners,  Pinturillo]</td>
      <td>24</td>
      <td>2017-08-16 21:15:00.000</td>
      <td>2017-08-17 00:30:00.000</td>
      <td>5823</td>
      <td>116</td>
      <td>5250</td>
      <td>4681</td>
      <td>8.29</td>
      <td>415.93</td>
      <td>210</td>
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
      <th>RowNum</th>
      <th>StartTime</th>
      <th>EndTime</th>
      <th>ViewsGained</th>
      <th>FollowersGained</th>
      <th>MaxViewers</th>
      <th>AverageViewers</th>
      <th>FollowersPerHour</th>
      <th>ViewsPerHour</th>
      <th>LengthMinutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>194</th>
      <td>August 10, 2015</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI]</td>
      <td>[Rebirth,  Nuclear Throne,  OlliOlli 2: Olliwo...</td>
      <td>429</td>
      <td>2015-08-10 22:15:00.000</td>
      <td>2015-08-11 01:00:00.000</td>
      <td>6630</td>
      <td>224</td>
      <td>7333</td>
      <td>6101</td>
      <td>18.67</td>
      <td>552.50</td>
      <td>180</td>
    </tr>
    <tr>
      <th>195</th>
      <td>July 30, 2015</td>
      <td>[Northernlion, RockLeeSmile]</td>
      <td>[Rebirth,  Duck Game (continued),  Speedrunners]</td>
      <td>435</td>
      <td>2015-07-30 22:15:00.000</td>
      <td>2015-07-31 01:00:00.000</td>
      <td>6798</td>
      <td>146</td>
      <td>5874</td>
      <td>4860</td>
      <td>12.17</td>
      <td>566.50</td>
      <td>180</td>
    </tr>
    <tr>
      <th>196</th>
      <td>July 30, 2015</td>
      <td>[Northernlion, RockLeeSmile]</td>
      <td>[Rebirth,  Duck Game (continued),  Speedrunners]</td>
      <td>436</td>
      <td>2015-07-30 00:30:00.000</td>
      <td>2015-07-30 01:00:00.000</td>
      <td>1550</td>
      <td>60</td>
      <td>4227</td>
      <td>3392</td>
      <td>12.00</td>
      <td>310.00</td>
      <td>75</td>
    </tr>
    <tr>
      <th>197</th>
      <td>July 29, 2015</td>
      <td>[Northernlion, MathasGames, RockLeeSmile, Alpa...</td>
      <td>[Rebirth &amp; audio problems,  Rocket League,  Nu...</td>
      <td>437</td>
      <td>2015-07-29 22:15:00.000</td>
      <td>2015-07-30 00:00:00.000</td>
      <td>5050</td>
      <td>169</td>
      <td>4308</td>
      <td>3749</td>
      <td>15.36</td>
      <td>459.09</td>
      <td>165</td>
    </tr>
    <tr>
      <th>198</th>
      <td>July 27, 2015</td>
      <td>[Northernlion, RockLeeSmile, JSmithOTI]</td>
      <td>[Rebirth,  Nuclear Throne,  Family Feud 2010 (...</td>
      <td>438</td>
      <td>2015-07-27 22:15:00.000</td>
      <td>2015-07-28 01:00:00.000</td>
      <td>8887</td>
      <td>181</td>
      <td>6658</td>
      <td>5865</td>
      <td>15.08</td>
      <td>740.58</td>
      <td>180</td>
    </tr>
  </tbody>
</table>
</div>






    199



Almost 200 lines that overlap. These are the videos we will get the comments for.


```python
combined.to_pickle('Pipeline\combined.pkl')
```
