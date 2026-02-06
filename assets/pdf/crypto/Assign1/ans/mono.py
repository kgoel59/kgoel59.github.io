import string
import random

def create_sub_map():
    # Create a list of uppercase letters
    letters = list(string.ascii_uppercase)
    
    # Shuffle the letters to create a random substitution mapping
    shuffled_letters = letters[:]
    random.shuffle(shuffled_letters)
    
    # Create the substitution map by pairing each letter with a letter from the shuffled list
    sub_map = dict(zip(letters, shuffled_letters))
    
    return sub_map

sub_map = create_sub_map()


print("".join([sub_map[i] for i in sub_map]))

def encode(submap, word):
    res = ""
    for c in word:
        res += submap[c]

    return res
caps = "KARAN".upper()
print(caps)
print(encode(sub_map, caps))