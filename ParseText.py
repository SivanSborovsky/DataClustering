import os
import re

data_folder = "Texts"
documents = []
#TODO: add all perek,pasuk and overall pasuk options
wrong_dict = {'א','ד','ג','ב','{ס}','{פ}','ה','{ש}','ו','ז'
              'ח','ט','י','כ','ל','מ','נ','ס','ע','פ','צ','ק','ר',
              'ש','ת','יא','יב','יג','יד','טה','טו','יז','יח','יט','כא','כב','כג','כד','כה',
            'כו','כז','כח','כט','לא','לב','לג','לד','לה','לו','לז','לח','לט','מא','מב','מג',
              'א,א', 'א,ב', 'א,ג', 'א,ד', 'א,ה', 'א,ו', 'א,ז', 'א,ח', 'א,ט', 'א,י', 'א,כ', 'א,יא', 'א,יב', 'א,יג', 'א,יד', 'א,טה',

              '',';','.',':'
              }
symbols = r";:,."
sentence_symbols = r";:."


"""
recieves a file and returns list of words in file
"""
def tokenize_file(filename):
    with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as file:
        document = file.read()
        documents.append(document)
    tokenized_words = []
    for sentence in documents:
        words = re.split(r"\s+|[" + symbols + r"]", sentence)
        words = [word for word in words if word]  # Remove empty words
        tokenized_words.extend(words)
    # print(tokenized_words)

    filtered_words = []
    for word in tokenized_words:
        if word not in wrong_dict:
            filtered_words.append(word)
    # print(filtered_words)
    return filtered_words

"""
recieves a list of sentences, deletes bad words and return fixed sentences
"""
def filter_sentences(sentences):
    filtered_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = []
        for word in words:
            if word not in wrong_dict:
                filtered_words.append(word)
        filtered_sentence = ' '.join(filtered_words)
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences


"""
recieves a file and returns it as a list of filtered sentences
"""
def parse_file(filename):
    with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as file:
        document = file.read()

    delimiter_pattern = r"[;:.]"
    sentences = re.split(delimiter_pattern, document)
    # print(sentences)
    filtered_sentences = filter_sentences(sentences)
    # print(filtered_sentences)
    return filtered_sentences
