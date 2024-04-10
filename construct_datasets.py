#%%
import pandas as pd 
import numpy as np 
import regex as re
import glob
from pynput import keyboard

files = glob.glob(r'C:\Users\trigou\Documents\python\code\AAA_1\*')

def get_text_with_word(f, regex_exp, window):

    with open(f, 'r') as t:

        text = t.read().lower()
        text = re.sub('[^\w\.]', ' ',text)
        text = re.sub('[0-9]', ' ', text)
        text = text.split()

        match = [True if (re.sub(regex_exp, '', sentence) != sentence) else False for sentence in text]
        match = [match[i] if (True not in match[i-window:i+window]) else True for i in range(window, len(match)-window) ]
        

        indexes = np.argwhere(match).flatten()

        split = [i if (indexes[i] != (indexes[i-1] + 1)) else np.nan for i in range(1, len(indexes))]
        split = [s for s in split if str(s) != 'nan']

        text = np.array(text[window:-window])[match]
        
        dataset = []
        for i in split:
            dataset.append(' '.join(text[i-window*2:i]))
            

        return dataset 
     

df = get_text_with_word(files[10], 'may|could|potentially', 5)



def on_release(key):

    key = str(key)

    if key == 'Key.right':
        with open('dataset_1.txt', 'a') as t:
            t.write(s + '\n')

        return False

    if key == 'Key.left':
        with open('dataset_0.txt', 'a') as t:
            t.write(s  + '\n')

        return False
    else:
        return False


for s in df:
    print(s, '\n')
    with keyboard.Listener(on_press=on_release) as listener:

        listener.join()


# %%
