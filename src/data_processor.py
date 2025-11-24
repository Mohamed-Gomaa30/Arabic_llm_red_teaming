import pandas as pd
import os
from typing import List, Dict
from configs.config_manager import ConfigManager
from pathlib import Path
from dotenv import load_dotenv
import mishkal.tashkeel
import logging
dotenv_path = Path(__file__).parent.parent / ".env" 
load_dotenv(dotenv_path)


logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.vocalizer = mishkal.tashkeel.TashkeelClass()
        self.data_config = config.get_data_config()
        self.setup_comprehensive_conversion_maps()
    
    def setup_comprehensive_conversion_maps(self):
        
        self.arabizi_no_numbers = {
            "أ": "a", "ا": "a", "آ": "a", "ؤ": "o'", "ئ": "'", "ء": "'", "إ": "e",
            "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh", "د": "d", 
            "ذ": "z", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "dh",
            "ط": "t", "ظ": "z", "ع": "'", "غ": "gh", "ف": "f", "ق": "q", "ك": "k",
            "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "t", "و": "w", "ي": "y", "ى": "a",
        }
        
        self.arabizi_with_numbers = {
            "أ": "'", "ا": "a", "إ": "i", "آ": "a'", "ؤ": "o'", "ئ": "'", "ء": "'",
            "ب": "b", "ت": "t", "ث": "th", "ج": "g", "ح": "7", "خ": "7'", "د": "d",
            "ذ": "th", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "9", "ض": "9'",
            "ط": "6", "ظ": "6'", "ع": "3", "غ": "3'", "ف": "f", "ق": "8", "ك": "k",
            "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "h'", "و": "w", "ي": "y", "ى": "a",
        }
        
        self.transliteration_map = {
            "أ": "a", "ا": "ā", "ؤ": "u'", "ئ": "'", "ء": "'", "إ": "i", "ب": "b",
            "ت": "t", "ث": "ṯ", "ج": "ǧ", "ح": "ḥ", "خ": "ẖ", "د": "d", "ذ": "ḏ",
            "ر": "r", "ز": "z", "س": "s", "ش": "š", "ص": "ṣ", "ض": "ḍ", "ط": "ṭ",
            "ظ": "ẓ", "ع": "ʿ", "غ": "ġ", "ف": "f", "ق": "q", "ك": "k", "ل": "l",
            "م": "m", "ن": "n", "ه": "h", "ة": "t", "و": "w", "ي": "y", "ى": "a",
        }
        
        self.common_words_no_numbers = {
            "ماهي": "mahya", " و": "wa", "من": "min", "الى": "ela", "في": "fi",
            "أعط": "a'ti", "اذكر": "uzkur", "عدد": "'adid", "التالية": "altaliya",
            "السابقة": "alsabiqah", "التي": "alati", "للتي": "lilati", "الذي": "alladhi",
            "للذي": "liladhi", "الذين": "alladhina", "للذين": "liladhina", "هذا": "hadha",
            "لهذا": "lhadha", "هذه": "hadhihi", "لهذه": "lhadhihi", "هذان": "hadhani",
            "لهذين": "lhadhaini", "هذين": "hadhaini", "هذي": "hadhi", "لهذي": "lhadhi",
        }
        
        self.common_words_with_numbers = {
            "ماهي": "mahya", " و": "wa", "من": "min", "الى": "ela", "في": "fe",
            "أعط": "a36i", "عدد": "3did", "التالية": "altaliya", "السابقة": "alsabi8at",
            "التي": "alati", "للتي": "lilati", "الذين": "allathina", "للذين": "lilathina",
            "هذا": "hatha", "لهذا": "lhatha", "هذه": "hathihi", "لهذه": "lhathihi",
            "هذان": "hathani", "لهذين": "lhathaini", "هذين": "hathaini", "هذي": "hathi",
            "لهذي": "lhathi",
        }
        
        self.transliteration_words = {
            "ماهي": "māhī", "من": "mn", "إلى": "āla", "في": "fī", "أعط": "ʾaʿt",
            "أذكر": "āḏkr", "عدد": "ʿdd", "التالية": "āltālīẗ", "السابقة": "ālsābqẗ",
            "التالي": "āltī", "للتي": "lltī", "الذي": "ālḏī", "للذي": "llḏī",
            "الذين": "ālḏīn", "للذين": "llḏīn", "هذا": "hḏā", "لهذا": "lhḏā",
            "هذه": "hḏh", "لهذه": "lhḏh", "هذان": "hḏān", "لهذين": "lhḏīn",
            "هذين": "hḏīn", "هذي": "hḏī", "لهذي": "lhḏī",
        }
    
    def convert_with_word_priority(self, text: str, char_map: Dict, word_map: Dict) -> str:
        words = str(text).split()
        converted_words = []
        
        for word in words:
            if word in word_map:
                converted_words.append(word_map[word])
            else:
                converted_word = ""
                for char in word:
                    converted_word += char_map.get(char, char)
                converted_words.append(converted_word)
        
        return " ".join(converted_words)
    
    def strip_existing_diacritics(self, text: str) -> str:
        import re
        diacritics_pattern = re.compile(r'[\u064B-\u065F\u0670]')
        return diacritics_pattern.sub('', text)
    
    def add_diacritics(self, text: str) -> str:
        try:
            clean_text = self.strip_existing_diacritics(str(text))
            
            vocalized_text = self.vocalizer.tashkeel(clean_text)
            
            if vocalized_text and vocalized_text.strip():
                return vocalized_text.strip()
            else:
                return clean_text
                    
        except Exception as e:
            print(f"Error in formatting using mashkal: {e}") 
            return str(text)
    
    def convert_to_arabizi_no_numbers(self, text: str) -> str:
        return self.convert_with_word_priority(text, self.arabizi_no_numbers, self.common_words_no_numbers)
    
    def convert_to_arabizi_with_numbers(self, text: str) -> str:
        return self.convert_with_word_priority(text, self.arabizi_with_numbers, self.common_words_with_numbers)
    
    def convert_to_transliteration(self, text: str) -> str:
        return self.convert_with_word_priority(text, self.transliteration_map, self.transliteration_words)
    
    def load_dataset(self, dataset_type: str = "both") -> pd.DataFrame:
        data_files = self.data_config['input_files']
        
        if dataset_type == "harmful":
            file_path = data_files['harmful']
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} records from {file_path}")
            
        elif dataset_type == "regional":
            file_path = data_files['regional']
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} records from {file_path}")
            
        else: 
            df1 = pd.read_csv(data_files['harmful'])
            df2 = pd.read_csv(data_files['regional'])
            df = pd.concat([df1, df2], ignore_index=True)
            print(f"Loaded {len(df)} records from both files")
        
        sample_size = self.data_config.get('sample_size', -1)
        if sample_size > 0 and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            print(f"A sample of size {len(df)} was taken")
        
        return df

    
    def process_data(self, dataset_type: str = "both") -> pd.DataFrame:
        df = self.load_dataset(dataset_type)
        
        print("Starting text conversion...")
        
        df['arabizi_no_numbers'] = df['Arabic'].apply(self.convert_to_arabizi_no_numbers)
        df['arabizi_with_numbers'] = df['Arabic'].apply(self.convert_to_arabizi_with_numbers)
        df['transliteration'] = df['Arabic'].apply(self.convert_to_transliteration)
        df['diacritized'] = df['Arabic'].apply(self.add_diacritics)
        
        print("Text conversion completed")
        print(f"Conversion examples:")
        print(f"   Arabic: {df['Arabic'].iloc[0][:50]}...")
        print(f"   Arabizi without numbers: {df['arabizi_no_numbers'].iloc[0][:50]}...")
        print(f"   Arabizi with numbers: {df['arabizi_with_numbers'].iloc[0][:50]}...")
        print(f"   Transliteration: {df['transliteration'].iloc[0][:50]}...")
        
        output_dir = self.data_config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'processed_{dataset_type}.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"Data saved to: {output_path}")
        
        return df