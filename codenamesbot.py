import tkinter as tk

import gensim
import inflect
import random
import numpy as np
HEIGHT = 700
WIDTH = 500
custom_blue = '#80b3ff'
model = gensim.models.KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000
)

with open("./words.txt") as f:
    words = f.readlines()

words = [w.strip() for w in words]

"""Generates team word lists and a random game board based on the word lists.

:param word_list: Codenames master word list, generated in block above
:return: 5x5 numpy board, red team words, blue team words, neutral words, assassin
:rtype: tuple
"""
"""Generates team word lists and a random game board based on the word lists.

:param word_list: Codenames master word list, generated in block above
:return: 5x5 numpy board, red team words, blue team words, neutral words, assassin
:rtype: tuple
"""
"""Generates team word lists and a random game board based on the word lists.

:param word_list: Codenames master word list, generated in block above
:return: 5x5 numpy board, red team words, blue team words, neutral words, assassin
:rtype: tuple
"""
def generate_board(word_list):
    used = set()
    red = []
    blue = []
    neutral = []
    assassin = []

    #Generate 9 random words for red team.
    while len(red) < 9:
        index = random.choice(range(len(word_list)))
        word = word_list[index]
        if index not in used:
            red.append(word)
            used.add(index)

    #Generate 8 random words for blue team.
    while len(blue) < 8:
        index = random.choice(range(len(word_list)))
        word = word_list[index]
        if index not in used:
            blue.append(word)
            used.add(index)
    
    #Generate 7 random neutral words.
    while len(neutral) < 7:
        index = random.choice(range(len(word_list)))
        word = word_list[index]
        if index not in used:
            neutral.append(word)
            used.add(index)
    
    #Generate assassin word.
    while not assassin:
        index = random.choice(range(len(word_list)))
        word = word_list[index]
        if index not in used:
            assassin.append(word)
            used.add(index)
    board = red + blue + neutral + assassin
    random.shuffle(board)
    board = np.reshape(board,(5,5))
    return board, red, blue, neutral, assassin

"""Guesses the most similar n words out of given words list based on given clue.

Threshold similarity for guessed words must be greater than 0.2

:param clue: given clue
:param words: given list of words to guess from
:param n: max number of words to guess
:return: list of length at most n of best guesses
"""
def guess(clue, words, n):
    poss = {}
    for w in words:
        poss[w] = model.similarity(clue, w)
    poss_lst = sorted(poss, key=poss.__getitem__, reverse=True)
    top_n = poss_lst[:n]
    return [w for w in top_n if poss[w] > 0.2]

"""Verifies that a clue is valid.

A clue (word2) is invalid if either word is a substring of the other, if there is an underscore
in word2, if word2 is the plural form of word1, or if the length of word2 is less than or equal to 2.
Uses inflect.engine() to check plurality.

:param word1: a word from the codenames list of words
:param word2: a model generated clue to be verified
:return: False if word2 is invalid, True if word2 is valid
"""
def clean_clue(word1, word2):
    engine = inflect.engine()
    word1 = word1.lower()
    word2 = word2.lower()
    return not (word1 in word2 or word2 in word1 or "_" in word2 or word2 == engine.plural(word1) or len(word2) <= 2)

"""Gives an optimal length clue based on current board state.

This function computes the similarity index between all pairs and triples of words.
Then, it iteratively computes an optimal clue based on the number of words left
and the max between the highest pair similarity and the highest triplet similarity.

:param words: list of words to generate a clue for
:param bad_words: list of words to avoid giving clues for
:return: tuple of optimal clue, the words intended to be guessed
"""
def give_clue(words, bad_words):
            
    # Correlates all possible pairs of words and store them in a dict of (word1,word2):similarity
    similarities = {}
    if len(words) >= 2:
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                similarities[(words[i], words[j])] = model.similarity(words[i], words[j])
    
    # Correlate all possible triplets of words and store them in a dict of (word1,word2,word3):similarity
    triple_similarities = {}
    if len(words) >= 3:
        seen = set()
        for w in words:
            for key in similarities.keys():
                z = key + (w,)
                if w not in key and tuple(sorted(z)) not in seen:
                    triple_similarities[z] = model.n_similarity([w], list(key))
                    seen.add(tuple(sorted(z)))
    
    # Loop until we find the optimal pair of words to give a clue for.
    while True:
        # If there is only 1 word left to guess or we run out of pairwise similarities,
        # set max_correlated_n to only word left
        if len(words) == 1 or not similarities:
            max_correlated_n = (words[0],)
        
        # If length is 2, set max_correlated_n to the max_correlated_pair
        elif len(words) >= 2:
            max_correlated_pair = max(similarities, key=similarities.get)
            max_correlated_n = max_correlated_pair
            
        # If length is 3, set max_correlated_n to:
        # max_correlated_triple if the triple similarity * 0.9 >= pair similarity
        # max_correlated_pair otherwise
        # 0.9 can be tuned based on experimental data
        if len(words) >= 3:
            max_correlated_triple = max(triple_similarities, key=triple_similarities.get)
            if triple_similarities[max_correlated_triple] * 0.9 >= similarities[max_correlated_pair]:
                max_correlated_n = max_correlated_triple
            else:
                max_correlated_n = max_correlated_pair
        
        print("Giving clue for:", max_correlated_n)
        c_words = list(max_correlated_n)
        
        # Find most similar words to words in max_correlated_n
        clues = model.most_similar(positive=c_words,topn=10, restrict_vocab=10000)
        
        # Clean the found similar words
        clues_dict = dict(clues)
        cleaned_clues = [c[0] for c in clues if all([clean_clue(w,c[0]) for w in c_words])]
        
        # Iterate until cleaned_clues is empty or we find an optimal clue
        while cleaned_clues:
            # Find best current clue
            possible_clue = max(cleaned_clues, key=lambda x: clues_dict[x])
            
            # If game is nearing end, skip filtering of clues
            if len(words) == len(max_correlated_n):
                return possible_clue, tuple(max_correlated_n)

            # Find most similar word to best current clue from bad_words
            enemy_match = model.most_similar_to_given(possible_clue, bad_words)
            
            # Calculate similarity between the two
            enemy_sim = model.similarity(enemy_match, possible_clue)
            
            # If enemy's word is greater in similarity than any of the words in max_correlated_pair,
            # remove the best current clue from cleaned_clues and continue iterating.
            # If not, return the current clue, as it is optimal.
            optimal = True
            for n in max_correlated_n:
                if enemy_sim >= model.similarity(n, possible_clue):
                    print("Foreign word " + enemy_match + " was too similar. Removing clue: " + possible_clue)
                    cleaned_clues.remove(possible_clue)
                    optimal = False
                    break
            
            if optimal:
                return possible_clue, tuple(max_correlated_n)
            
        # All the enemy's clues were atleast more similar than one of the words in max_correlated_pair,
        # so pop max_correlated_n from corresponding similarities dict and continue iterating.
        print("Too many enemy correlations. Removing ", max_correlated_n)
        if len(max_correlated_n) == 2:
            similarities.pop(max_correlated_n) 
        elif len(max_correlated_n) == 3:
            triple_similarities.pop(max_correlated_n)

class CodeNames():
	def __init__(self):
		def start_game(type):
			if type == 0:
				agent_begin()
			else:
				spy_begin()
		def agent_begin():
			startmenu.destroy()
			label.destroy()
			frame.destroy()
			agent_button.destroy()
			spy_button.destroy()
			red_score = 0
			blue_score = 0
			curr_player = 0
			board, red, blue, neutral, assassin = generate_board(words)
			initial_clue = give_clue(red, blue + assassin + neutral)
			num_guesses = len(initial_clue[1])
			
			game_state = tk.Frame(self.root, bg = custom_blue, bd = 10)
			game_state.place(relx = 0.5, rely = .05, relwidth = .9, relheight = 0.5, anchor = 'n')
			
			scoreboard = tk.Frame(self.root, bg = custom_blue, bd = 5)
			scoreboard.place(relx = 0.87, rely = 0.8, relwidth = .15, relheight = 0.1, anchor = 'n')
			red_team = tk.Label(scoreboard, text = "Red: " + str(red_score), bg = 'white', font = ('Courier', 10))
			red_team.place(relx = 0, rely = 0, relwidth = 1, relheight = 0.45)			
			blue_team = tk.Label(scoreboard, text = "Blue: " + str(blue_score), bg = 'white', font = ('Courier', 10))
			blue_team.place(relx = 0, rely = .55, relwidth = 1, relheight = 0.45)

			hint_frame = tk.Frame(self.root, bg = custom_blue, bd = 7)
			hint_frame.place(relx = 0.5, rely = .65, relwidth = .9, relheight = .1, anchor = 'n')
			current_hint = tk.Message(hint_frame, text = "Clue is: " + initial_clue[0], bg = 'white', font = ('Courier', 14), aspect = 1000)
			current_hint.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)
			
			message_frame = tk.Frame(self.root, bg = custom_blue, bd = 7)
			message_frame.place(relx = 0.4, rely = 0.8, relwidth = .7, relheight = 0.1, anchor = 'n')
			current_turn = tk.Message(message_frame, text = "Welcome to CodeNames! It's Red's turn with " + str(num_guesses) + " guesses remaining.", bg = 'white', font = ('Courier', 10), aspect = 800)
			current_turn.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)

			pass_frame = tk.Frame(self.root, bg = custom_blue, bd = 5)
			pass_frame.place(relx = 0.5, rely = .58, relwidth = .2, relheight = .05, anchor = 'n')
			pass_turn = tk.Button(pass_frame, text = 'Pass Turn', bg = 'white', font = ('Courier', 10), command = lambda: skip_turn())
			pass_turn.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)
			
			


			bwc = {} #button_word_connector
			seen_words = set()
			def button_action(word):
				nonlocal red_score
				nonlocal blue_score
				nonlocal curr_player
				nonlocal red_team
				nonlocal blue_team
				nonlocal current_turn
				nonlocal num_guesses
				nonlocal seen_words
				if word in seen_words:
					return
				seen_words.add(word)
				if word in red and curr_player == 0 and num_guesses > 0:
					red_score += 1
					red.remove(word)
					num_guesses -= 1
					bwc.get(word).configure(highlightbackground = '#b32400', highlightthickness = 50)
					red_team.configure(text = "Red: " + str(red_score))
					current_turn.configure(text = "Red Team has " + str(num_guesses) + " guesses left.")
					if num_guesses == 0:
						next_clue = give_clue(blue, red + neutral + assassin)
						num_guesses = len(next_clue[1])
						curr_player = 1 - curr_player
						current_turn.configure(text = "It is now Blue's turn with " + str(num_guesses) + " guesses remaining.")
						current_hint.configure(text = "Clue is " + next_clue[0])
				elif word in blue and curr_player == 1 and num_guesses > 0:
					blue_score += 1
					blue.remove(word)
					num_guesses -= 1
					bwc.get(word).configure(highlightbackground = '#0066cc', highlightthickness = 50)
					blue_team.configure(text = "Blue: " + str(blue_score))
					current_turn.configure(text = "Blue Team has " + str(num_guesses) + " guesses left.")
					if num_guesses == 0:
						next_clue = give_clue(red, blue + neutral + assassin)
						num_guesses = len(next_clue[1])
						curr_player = 1 - curr_player
						current_turn.configure(text = "It is now Red's turn with " + str(num_guesses) + " guesses remaining.")
						current_hint.configure(text = "Clue is " + next_clue[0])
				elif word in red and curr_player == 1:
					red_score += 1
					red.remove(word)
					bwc.get(word).configure(highlightbackground = '#b32400', highlightthickness = 50)
					red_team.configure(text = "Red: " + str(red_score))
					next_clue = give_clue(red, blue + neutral + assassin)
					num_guesses = len(next_clue[1])
					curr_player = 1 - curr_player
					current_turn.configure(text = "You guessed Red Team's word! It's Red Team's turn with " + str(num_guesses) + " guesses remaining.")
					current_hint.configure(text = "Clue is " + next_clue[0])
				elif word in blue and curr_player == 0:
					blue_score += 1
					blue.remove(word)
					bwc.get(word).configure(highlightbackground = '#0066cc', highlightthickness = 50)
					blue_team.configure(text = "Blue: " + str(blue_score))
					next_clue = give_clue(blue, red + neutral + assassin)
					num_guesses = len(next_clue[1])
					curr_player = 1 - curr_player
					current_turn.configure(text = "You guessed Blue Team's word! It's Blue Team's turn with " + str(num_guesses) + " guesses remaining.")
					current_hint.configure(text = "Clue is " + next_clue[0])
				elif word in assassin:
					print("Game Over! You picked the assassin word. Player", 1 - curr_player, "wins!")
					bwc.get(word).configure(highlightbackground = 'black', highlightthickness = 50)
					victory(0)
				else: #neutral case
					neutral.remove(word)
					if curr_player == 0:
						next_clue = give_clue(blue, red + neutral + assassin)
						num_guesses = len(next_clue[1])
						current_turn.configure(text = "You guessed a neutral word! It's Blue Team's turn with " + str(num_guesses) + " guesses remaining.")
						current_hint.configure(text = "Clue is " + next_clue[0])

					elif curr_player == 1:
						next_clue = give_clue(red, blue + neutral + assassin)
						num_guesses = len(next_clue[1])
						current_turn.configure(text = "You guessed a neutral word! It's Red Team's turn with " + str(num_guesses) + " guesses remaining.")
						current_hint.configure(text = "Clue is " + next_clue[0])
					curr_player = 1 - curr_player
					bwc.get(word).configure(highlightbackground = '#bfbfbf', highlightthickness = 50)
				if red_score == 9:
					victory(red_score)
					current_hint.configure(text = "Congratulations! Red Team Wins!")
				if blue_score == 8:
					victory(blue_score)
					current_hint.configure(text = "Congratulations! Blue Team Wins!")

			def skip_turn():
				nonlocal curr_player
				nonlocal red_team
				nonlocal blue_team
				nonlocal current_turn
				nonlocal num_guesses
				if curr_player == 0:
					next_clue = give_clue(blue, red + neutral + assassin)
					num_guesses = len(next_clue[1])
					current_turn.configure(text = "You passed! It's Blue Team's turn with " + str(num_guesses) + " guesses remaining.")
					current_hint.configure(text = "Clue is " + next_clue[0])
				else:
					next_clue = give_clue(red, blue + neutral + assassin)
					num_guesses = len(next_clue[1])
					current_turn.configure(text = "You passed! It's Red Team's turn with " + str(num_guesses) + " guesses remaining.")
					current_hint.configure(text = "Clue is " + next_clue[0])
				curr_player = 1 - curr_player
			for i in range(5):
				for j in range(5):
					s = board[i][j]
					game_button = tk.Button(game_state, text = board[i][j], highlightbackground = 'white', highlightthickness = 50, font = ('Courier New', 10), command = lambda v=s: button_action(v))
					game_button.place(relx = i/5, rely = j/5, relwidth = 0.2, relheight = 0.2)
					bwc[board[i][j]] = game_button
			def victory(score):
				victory_frame = tk.Frame(self.root, bg = 'white')
				victory_frame.place(relx = .5, rely = 0, relwidth = 1, relheight = 1, anchor = 'n')
				if score == 9:
					victory_title = tk.Label(victory_frame, bg = '#ffcccb', font = ('Courier', 15), text = "Congratulations! Red Team Wins!")
					victory_title.place(relx = 0, rely = 0, relwidth =1, relheight = 1)
					exit_button = tk.Button(victory_frame, text = 'Quit', bg = 'white', font = ('Courier', 10), command = lambda: self.root.destroy())
					exit_button.place(relx = 0.5, rely = 0.7, relwidth = .3, relheight = 0.1, anchor = 'n')
				elif score == 8:
					victory_title = tk.Label(victory_frame, bg = custom_blue, font = ('Courier', 15), text = "Congratulations! Blue Team Wins!")
					victory_title.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)
					exit_button = tk.Button(victory_frame, text = 'Quit', bg = 'white', font = ('Courier', 10), command = lambda: self.root.destroy())
					exit_button.place(relx = 0.5, rely = 0.7, relwidth = .3, relheight = 0.1, anchor = 'n')
				else:
					defeat_title = tk.Label(victory_frame, bg = '#bfbfbf', font = ('Courier', 15), text = "Game Over! You picked the assassin word.")
					defeat_title.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)
					exit_button = tk.Button(victory_frame, text = 'Quit', bg = 'white', font = ('Courier', 10), command = lambda: self.root.destroy())
					exit_button.place(relx = 0.5, rely = 0.7, relwidth = .3, relheight = 0.1, anchor = 'n')


		def spy_begin():
			startmenu.destroy()
			label.destroy()
			frame.destroy()
			agent_button.destroy()
			spy_button.destroy()
			red_score = 0
			blue_score = 0
			curr_player = 0
			board, red, blue, neutral, assassin = generate_board(words)
			game_state = tk.Frame(self.root, bg = custom_blue, bd = 10)
			game_state.place(relx = 0.5, rely = .05, relwidth = .9, relheight = 0.5, anchor = 'n')
			seen_words = set()
			def computer_guess():
				nonlocal blue_score
				nonlocal red_score
				nonlocal curr_player
				nonlocal current_turn
				nonlocal seen_words
				nonlocal red_team
				nonlocal blue_team
				clue = current_hint.get()
				if clue not in model.vocab:
					current_turn.configure(text = "Invalid clue. Please try again.")
					return
				#if type()
				num_guesses = current_num.get()
				num_guesses = int(num_guesses)
				guess_list = guess(clue, red + blue + neutral + assassin, int(num_guesses))
				if not guess_list:
					if curr_player == 0:
						current_turn.configure(text = "It's Blue's turn. Input clue on left and number of guesses on right.")
					else:
						current_turn.configure(text = "It's Red's turn. Input clue on left and number of guesses on right.")
					curr_player = 1 - curr_player
					return
				for g in guess_list:
					if g in red:
						bwc.get(g).configure(bg = 'black')
						red_score += 1
						red_team.configure(text = "Red: " + str(red_score))
						num_guesses -= 1
						if red_score == 9:
							victory(red_score)
						red.remove(g)
						if curr_player == 1:
							curr_player = 1 - curr_player
							current_turn.configure(text = "It's Red's turn. Input clue on left and number of guesses on right.")
							break
						if num_guesses == 0:
							curr_player = 1 - curr_player
							current_turn.configure(text = "It's Blue's turn. Input clue on left and number of guesses on right.")
							break

					elif g in blue:
						bwc.get(g).configure(bg = 'black')
						blue_score += 1
						blue_team.configure(text = "Blue: " + str(blue_score))
						num_guesses -= 1
						if blue_score == 8:
							victory(blue_score)
						blue.remove(g)
						if num_guesses == 0:
							curr_player = 1 - curr_player
							current_turn.configure(text = "It's Red's turn. Input clue on left and number of guesses on right.")
							break
						if curr_player == 0:
							curr_player = 1 - curr_player
							current_turn.configure(text = "It's Blue's turn. Input clue on left and number of guesses on right.")
							break
					elif g in neutral:
						bwc.get(g).configure(bg = 'black')
						neutral.remove(g)
						if curr_player == 0:
							current_turn.configure(text = "It's Blue's turn. Input clue on left and number of guesses on right.")
						else:
							current_turn.configure(text = "It's Red's turn. Input clue on left and number of guesses on right.")
						curr_player = 1 - curr_player
						break
					else:
						victory(0)

		

				


			scoreboard = tk.Frame(self.root, bg = custom_blue, bd = 5)
			scoreboard.place(relx = 0.87, rely = 0.8, relwidth = .15, relheight = 0.1, anchor = 'n')
			red_team = tk.Label(scoreboard, text = "Red: " + str(red_score), bg = 'white', font = ('Courier', 10))
			red_team.place(relx = 0, rely = 0, relwidth = 1, relheight = 0.45)			
			blue_team = tk.Label(scoreboard, text = "Blue: " + str(blue_score), bg = 'white', font = ('Courier', 10))
			blue_team.place(relx = 0, rely = .55, relwidth = 1, relheight = 0.45)

			hint_frame = tk.Frame(self.root, bg = custom_blue, bd = 7)
			hint_frame.place(relx = 0.4, rely = .65, relwidth = .7, relheight = .1, anchor = 'n')
			current_hint = tk.Entry(hint_frame, text = "Give clue: ", bg = 'white', font = ('Courier', 14))
			current_hint.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)

			num_frame = tk.Frame(self.root, bg = custom_blue, bd = 7)
			num_frame.place(relx = 0.85, rely = .65, relwidth = .1, relheight = .1, anchor = 'n')
			current_num = tk.Entry(num_frame, text = "Give number: ", bg = 'white', font = ('Courier', 14))
			current_num.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)
			
			message_frame = tk.Frame(self.root, bg = custom_blue, bd = 7)
			message_frame.place(relx = 0.4, rely = 0.8, relwidth = .7, relheight = 0.1, anchor = 'n')
			current_turn = tk.Message(message_frame, text = "Welcome to CodeNames! It's Red's turn. Input clue on left and number of guesses on right.", bg = 'white', font = ('Courier', 10), aspect = 800)
			current_turn.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)

			pass_frame = tk.Frame(self.root, bg = custom_blue, bd = 5)
			pass_frame.place(relx = 0.5, rely = .58, relwidth = .2, relheight = .05, anchor = 'n')
			pass_turn = tk.Button(pass_frame, text = 'Submit Clue', bg = 'white', font = ('Courier', 10), command = lambda: computer_guess())
			pass_turn.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)
			
			bwc = {}
			for i in range(5):
				for j in range(5):
					if board[i][j] in red:
						game_button = tk.Button(game_state, text = board[i][j], highlightbackground = '#b32400', font = ('Courier New', 10), highlightthickness=50)
					elif board[i][j] in blue:
						game_button = tk.Button(game_state, text = board[i][j], highlightbackground = '#0066cc', font = ('Courier New', 10), highlightthickness=50)
					elif board[i][j] in neutral:
						game_button = tk.Button(game_state, text = board[i][j], highlightbackground = '#bfbfbf', font = ('Courier New', 10), highlightthickness=50)
					else:
						game_button = tk.Button(game_state, text = board[i][j], highlightbackground = '#404040', font = ('Courier New', 10), highlightthickness=50)
					game_button.place(relx = i/5, rely = j/5, relwidth = 0.2, relheight = 0.2)
					bwc[board[i][j]] = game_button

			def victory(score):
				victory_frame = tk.Frame(self.root, bg = 'white')
				victory_frame.place(relx = .5, rely = 0, relwidth = 1, relheight = 1, anchor = 'n')
				if score == 9:
					victory_title = tk.Label(victory_frame, bg = '#ffcccb', font = ('Courier', 15), text = "Congratulations! Red Team Wins!")
					victory_title.place(relx = 0, rely = 0, relwidth =1, relheight = 1)
					exit_button = tk.Button(victory_frame, text = 'Quit', bg = 'white', font = ('Courier', 10), command = lambda: self.root.destroy())
					exit_button.place(relx = 0.5, rely = 0.7, relwidth = .3, relheight = 0.1, anchor = 'n')
				elif score == 8:
					victory_title = tk.Label(victory_frame, bg = custom_blue, font = ('Courier', 15), text = "Congratulations! Blue Team Wins!")
					victory_title.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)
					exit_button = tk.Button(victory_frame, text = 'Quit', bg = 'white', font = ('Courier', 10), command = lambda: self.root.destroy())
					exit_button.place(relx = 0.5, rely = 0.7, relwidth = .3, relheight = 0.1, anchor = 'n')
				else:
					defeat_title = tk.Label(victory_frame, bg = '#bfbfbf', font = ('Courier', 15), text = "Game Over! You picked the assassin word.")
					defeat_title.place(relx = 0, rely = 0, relwidth = 1, relheight = 1)
					exit_button = tk.Button(victory_frame, text = 'Quit', bg = 'white', font = ('Courier', 10), command = lambda: self.root.destroy())
					exit_button.place(relx = 0.5, rely = 0.7, relwidth = .3, relheight = 0.1, anchor = 'n')
		
		
		self.root = tk.Tk()
		self.root.resizable(0,0)
		startpage = tk.Canvas(self.root, height = HEIGHT, width = WIDTH)
		startpage.pack()
		background_image = tk.PhotoImage(file = 'planets2.png')
		background_label = tk.Label(self.root, image = background_image)
		background_label.place(relwidth=1, relheight = 1)
		frame = tk.Frame(self.root, bg = custom_blue, bd = 5)
		frame.place(relx = 0.5, rely = 0.1, relwidth = 0.55, relheight = 0.1, anchor = 'n')

		label = tk.Label(frame, text = "Codenames Bot", font = ('Courier', 24), bg = 'white')
		label.place(relwidth = 1, relheight = 1)

		startmenu = tk.Frame(self.root, bg = custom_blue, bd = 5)
		startmenu.place(relx = 0.5, rely = 0.5, relwidth = 0.35, relheight = 0.2, anchor = 'n')

		agent_button = tk.Button(startmenu, text = 'Play as Agent', bg = 'white', font = ('Courier', 10), command = lambda: start_game(0))
		agent_button.place(relx = 0, rely = 0, relwidth = 1, relheight = 0.475)
		
		spy_button = tk.Button(startmenu, text = 'Play as Spymaster', bg = 'white', font = ('Courier', 10), command = lambda: start_game(1))
		spy_button.place(relx = 0, rely = 0.525, relwidth = 1, relheight = 0.475)
		
		self.root.mainloop()
app = CodeNames()
