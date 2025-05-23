from __future__ import unicode_literals

import pandas as pd
import numpy as np
import os
import re
from config import DATA_ROOT, AGGREGATED_REL_PATH, INDIVIDUAL_REL_PATH

def remove_hash_mention(text):
    pattern  = re.compile(r'@([a-zA-Z0-9]{1,15})', re.I)
    if bool(pattern.search(text)) == True:
        words = text.split()
        new_text = ''
        for word in words:
            if word.startswith('#'):
                word=replace_names(word)
                word = re.sub('#', '', word)
                new_text += ' '+ word
            else:
                new_text += ' '+ word
            text = new_text.lstrip()
    else:
        pass
    return text

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    urls = url_pattern.findall(text)
    replacement_url = '[URL]'
    for url in urls:
        text = text.replace(url, replacement_url)
    return text

def replace_emojis(text):
    ## replacing some usual text emoticons to emojis before the next step
    ## this has to be done before the next step because the emojis would be counted then
    text = text.replace(r":)","🙂").replace(r":-)","🙂")
    text = text.replace(r":(","🙁").replace(r":-(","🙁")
    text = text.replace(r"xD", "😆").replace(r"XD", "😆").replace(r"xd", "😆")
    text = text.replace(r":'‑(", '😢').replace(":'(", "😢")
    text = text.replace(r":D", "😃").replace(":-D", "😃")
    text = text.replace(r";)", "😉").replace(";-)", "😉")
    return text

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

mention_pattern = re.compile('([^a-zA-Z0-9]|^)@\S+', flags=re.UNICODE) 
hashtag_pattern = re.compile('([^a-zA-Z0-9]|^)#\S+', flags=re.UNICODE)

def detweet(text):
    return re.sub(mention_pattern, '@MENTION', # for the remaining mentions
                re.sub(hashtag_pattern, '#[HASH]', # for the remaining hashtags
                    re.sub(emoji_pattern, '', # for remaining emojis
                            text)))


def normalize(text):
    return re.sub(r"\s+", " ", #remove extra spaces
                  re.sub(r'[^a-zA-Z0-9]', ' ', #remove non alphanumeric, incl punctuation
                         text)).lower().strip() #lowercase and strip

def fix_string(text):
    # remove extra spacing from text
    text = re.sub(' +', ' ', text)

    # countering hashtags with words -- replaces any normal words containing a hashtag in the beginning with the actual word itself
    # e.g., #redhead --> redhead
    textList = text.split()
    x = ''
    for i in textList:
        if (i[0] == '#'):
            if (i[1:] in words.words()):
                i = i.replace('#', '')
        x += ' '+ i
    text = x.lstrip()

    ## replace some general text alterations which would otherwise go undetected
    text = text.replace(r"&amp;", "and")

    ## expanding on contractions
    text = contractions.fix(text)

    # removing unicode emojis from the text
    # at the end for removing any trailing emoji patterns which add no value to the text
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text


def cleaning(text):
    if (type(text)==str) or (type(text)==unicode):
        return fix_string(normalize(detweet(remove_urls(remove_hash_mention(replace_emojis(text))))))
    else:
        return text


if __name__=='__main__':
    text = None
    with codecs.open(os.path.join(DATA_ROOT, ), encoding='utf8') as file_handle:
        text = file_handle.readlines()