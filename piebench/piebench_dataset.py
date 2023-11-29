import numpy as np
import PIL.Image
import pandas as pd
import re
import os

from torch.utils.data import Dataset

def transform_sentences(sentence1, sentence2):
    """
    Transform a pair of sentences by keeping only the central different words.

    :param sentence1: First sentence as a string.
    :param sentence2: Second sentence as a string.
    :return: A tuple of strings with only the central different words from both sentences.
    """
    # Split the sentences into words
    words1, words2 = sentence1.split(), sentence2.split()

    # Find the starting index of difference
    start_idx = 0
    for w1, w2 in zip(words1, words2):
        if w1 != w2:
            break
        start_idx += 1

    # Find the ending index of difference for each sentence
    end_idx1, end_idx2 = len(words1), len(words2)
    for i in range(1, min(len(words1), len(words2)) + 1):
        if words1[-i] != words2[-i]:
            end_idx1, end_idx2 = len(words1) - i + 1, len(words2) - i + 1
            break

    # Extract the different parts
    diff_part1 = ' '.join(words1[start_idx:end_idx1])
    diff_part2 = ' '.join(words2[start_idx:end_idx2])

    return (diff_part1, diff_part2)

# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def rle2mask(mask_rle, shape=(512,512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
    

common_words = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", 
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", 
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", 
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", 
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take", 
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", 
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also", 
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", 
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", 
    "is", "am", "are", "was", "were", "been", "has", "had", "did", "doing", 
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", 
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", 
    "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", 
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", 
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", 
    "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", 
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", 
    "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", 
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", 
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", 
    "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", 
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", 
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", 
    "you've", "your", "yours", "yourself", "yourselves"
]
def clean_common_word(l):
    return [w for w in l if w not in common_words]

class PieBenchDataset(Dataset):
    def __init__(self, json_path, images_path, side=512, transform=None):
        self.side = 512
        self.annotations = self._read_annotations(json_path)
        self.TASKS_IDS = self.annotations.editing_type_id.unique().tolist()
        self.root_dir = images_path
        self.transform = transform
    
    @staticmethod
    def _read_annotations(json_path):
        # Read annotations
        # columns: ['image_path', 'original_prompt', 'editing_prompt',
        #           'editing_instruction', 'editing_type_id', 'blended_word', 'mask']
        df = pd.read_json(json_path).T
        
        # Chand edit type id to int
        df['editing_type_id'] = df['editing_type_id'].astype(int)
        
        # Fix input
        df.loc[df.blended_word == "And chair rock", 'blended_word'] = "chair rock"
        df.loc[df.image_path == '0_random_140/000000000117.jpg',['original_prompt','editing_prompt']] = 'a [pink flower with yellow] center in the middle','a [blue flower with red] center in the middle'
        
        # Extract words
        regex = r'\[([^]]*)\]'
        l = [(re.findall(regex, i),re.findall(regex, o)) for i,o in zip(df.original_prompt,df.editing_prompt) ]
        df[['original_words','editing_words']] = [(";".join(e1),";".join(e2)) for (e1,e2) in l]
        # remove "," from the words
        df['original_words'] = df['original_words'].str.replace(',','')
        df['editing_words'] = df['editing_words'].str.replace(',','')
        # apply transform_sentences_v2 to the words
        df[['original_words_clean','editing_words_clean']] = [transform_sentences(e1,e2) for (e1,e2) in zip(df.original_words,df.editing_words)]
        
        # remove "[" and "]" from "original_prompt" and "editing_prompt"
        df['original_prompt_clean'] = df['original_prompt'].str.replace('[','')
        df['original_prompt_clean'] = df['original_prompt_clean'].str.replace(']','')
        df['editing_prompt_clean'] = df['editing_prompt'].str.replace('[','')
        df['editing_prompt_clean'] = df['editing_prompt_clean'].str.replace(']','')
        df['editing_prompt_clean']

        # add "is_cross" column
        df['is_cross'] = (df['original_words_clean'].str.len() != 0) & (df['editing_words_clean'].str.len() != 0)
        
        return df
    
    def __len__(self):
        return len(self.annotations)
    
    def type_id_index(self, id):
        return self.annotations[self.annotations.editing_type_id == id].index
        
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        
        # Load image
        image_path = row.image_path
        image = PIL.Image.open(os.path.join(self.root_dir,image_path)).convert('RGB').resize((self.side,self.side))
        
        # Blend words
        blended_word = row.blended_word.split()
        blend = blended_word if len(blended_word) > 0 else None
        
        # Replace words
        
        original_words = row.original_words_clean.split(';') if len(row.original_words_clean) > 0 else []
        editing_words = row.editing_words_clean.split(';') if len(row.editing_words_clean) > 0 else []
        replace = (original_words, editing_words) \
            if (len(original_words) > 0 and len(editing_words) > 0) else None
        
        # Target words
        editing_words_clean = clean_common_word([w for l in row.editing_words_clean.split(';') for w in l.split()])
        
        
        # Mask
        mask = rle2mask(row['mask'], shape=(self.side,self.side))[...,None]
        
        # Prompts
        original_prompt = row.original_prompt_clean
        editing_prompt = row.editing_prompt_clean
        
        sample = {
            'editing_type_id': row.editing_type_id,
            'image': image,
            'blend': blend,
            'replace': replace,
            'editing_words': editing_words_clean,
            'mask': mask,
            'original_prompt': original_prompt,
            'editing_prompt': editing_prompt,
            'image_path': image_path,
        }
        return sample

        
    