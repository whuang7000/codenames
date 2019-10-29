# Codenames Bot

This is a Codenames bot made by William Huang, Jonathan Sham, Michael Li, and Franklin Chian. It utilizes a pre-trained, deep learning word embedding model to predict relationships between words on the Codenames board. The bot can either give hints to the 2 users playing the game, or guess words from the board after being given hints from the user.

>Codenames is a game of guessing which code names (words) in a set are related to a hint-word given by another player.
>
>Players split into two teams: red and blue. One player of each team is selected as the team's spymaster; the others are field operatives.
>
>Twenty-five code name cards, each bearing a word, are laid out in a 5×5 rectangular grid, in random order. A number of these words represent red agents, a number represent blue agents, one represents an assassin, and the others represent innocent bystanders.
>
>The teams' spymasters are given a randomly-dealt map card showing a 5×5 grid of 25 squares of various colors, each corresponding to one of the code name cards on the table. Teams take turns. On each turn, the appropriate spymaster gives a verbal hint about the words on the respective cards. Each hint may only consist of one single word and a number. The spymaster gives a hint that is related to as many of the words on his/her own agents' cards as possible, but not to any others – lest they accidentally lead their team to choose a card representing an innocent bystander, an opposing agent, or the assassin.
>
>The hint's word can be chosen freely, as long as it is not (and does not contain) any of the words on the code name cards still showing at that time. Code name cards are covered as guesses are made.
>
>The hint's number tells the field operatives how many words in the grid are related to the word of the clue. It also determines the maximum number of guesses the field operatives may make on that turn, which is the hint's number plus one. Field operatives must make at least one guess per turn, risking a wrong guess and its consequences. They may also end their turn voluntarily at any point thereafter.
>
>After a spymaster gives the hint with its word and number, their field operatives make guesses about which code name cards bear words related to the hint and point them out, one at a time. When a code name card is pointed out, the spymaster covers that card with an appropriate identity card – a blue agent card, a red agent card, an innocent bystander card, or the assassin card – as indicated on the spymasters' map of the grid. If the assassin is pointed out, the game ends immediately, with the team who identified him losing. If an agent of the other team is pointed out, the turn ends immediately, and that other team is also one agent closer to winning. If an innocent bystander is pointed out, the turn simply ends.
>
>The game ends when all of one team's agents are identified (winning the game for that team), or when one team has identified the assassin (losing the game).

## Usage

The following packages are required for running this program. We apologize in advance for not using Piplock.
Required Commands:<br/>
```
pip install gensim
pip install inflect
pip install numpy
```


You must also download Google's DataSetGoogleNews at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?fbclid=IwAR2cGOoPhXxpTx0GSKLOp8xCbAVFWXcofbZ16NSeyZZ9rD0AOmFXr8M95bU. This file is 1.5 GB and contains Google's dataset of millions of sentences. Place this file in the root directory of this project.

To run the program, run `python codenamesbot.py`

Due to limitations of tkinter on MacOS, we recommend the use of Windows in running this program, as our UI choices were designed in a Windows environment. Nevertheless, our program will still run smoothly on MacOS.

![](spymaster.png)

