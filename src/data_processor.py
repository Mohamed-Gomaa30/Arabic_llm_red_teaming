import pandas as pd
import os
from typing import List, Dict
from config.config_manager import ConfigManager

class DataProcessor:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.data_config = config.get_data_config()
        self.setup_conversion_maps()
    
    def setup_conversion_maps(self):
        """إعداد قوائم التحويل"""
        # عربيزي بأرقام
        self.arabizi_numbers = {
            "أ": "'", "ا": "a", "إ": "i", "آ": "a'", "ء": "'",
            "ب": "b", "ت": "t", "ث": "th", "ج": "g", "ح": "7", "خ": "7'",
            "د": "d", "ذ": "th", "ر": "r", "ز": "z", "س": "s", "ش": "sh",
            "ص": "9", "ض": "9'", "ط": "6", "ظ": "6'", "ع": "3", "غ": "3'",
            "ف": "f", "ق": "8", "ك": "k", "ل": "l", "م": "m", "ن": "n",
            "ه": "h", "ة": "h'", "و": "w", "ي": "y", "ى": "a",
        }
        
        # تحويل صوتي
        self.transliteration_map = {
            "أ": "a", "ا": "a", "إ": "i", "ب": "b", "ت": "t", "ث": "th",
            "ج": "j", "ح": "h", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r",
            "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t",
            "ظ": "z", "ع": "'", "غ": "gh", "ف": "f", "ق": "q", "ك": "k",
            "ل": "l", "م": "m", "ن": "n", "ه": "h", "ة": "t", "و": "w",
            "ي": "y", "ى": "a",
        }
    
    def load_dataset(self, dataset_type: str = "both") -> pd.DataFrame:
        """تحميل البيانات المحددة"""
        data_files = self.data_config['input_files']
        
        if dataset_type == "harmful":
            file_path = data_files['harmful']
            df = pd.read_csv(file_path)
            print(f"✅ تم تحميل {len(df)} سجل من {file_path}")
            
        elif dataset_type == "regional":
            file_path = data_files['regional']
            df = pd.read_csv(file_path)
            print(f"✅ تم تحميل {len(df)} سجل من {file_path}")
            
        else:  # both
            df1 = pd.read_csv(data_files['harmful'])
            df2 = pd.read_csv(data_files['regional'])
            df = pd.concat([df1, df2], ignore_index=True)
            print(f"✅ تم تحميل {len(df)} سجل من كلا الملفين")
        
        # أخذ عينة إذا مطلوب
        sample_size = self.data_config.get('sample_size', -1)
        if sample_size > 0 and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            print(f"🔬 تم أخذ عينة حجمها {len(df)}")
        
        return df
    
    def convert_to_arabizi(self, text: str) -> str:
        """تحويل إلى عربيزي بأرقام"""
        result = ""
        for char in str(text):
            result += self.arabizi_numbers.get(char, char)
        return result
    
    def convert_to_transliteration(self, text: str) -> str:
        """تحويل إلى transliteration"""
        result = ""
        for char in str(text):
            result += self.transliteration_map.get(char, char)
        return result
    
    def add_diacritics(self, text: str) -> str:
        """إضافة تشكيل مبسط"""
        # هذه نسخة مبسطة - يمكن استبدالها بمكتبة متخصصة
        return str(text)  # في الإنتاج الحقيقي، أضف التشكيل هنا
    
    def process_data(self, dataset_type: str = "both") -> pd.DataFrame:
        """معالجة البيانات الرئيسية"""
        df = self.load_dataset(dataset_type)
        
        print("🔄 بدء تحويل النصوص...")
        
        # تطبيق التحويلات
        df['arabizi'] = df['Arabic'].apply(self.convert_to_arabizi)
        df['transliteration'] = df['Arabic'].apply(self.convert_to_transliteration)
        df['diacritized'] = df['Arabic'].apply(self.add_diacritics)
        
        print("✅ اكتمل تحويل النصوص")
        
        # حفظ البيانات المحولة
        output_dir = self.data_config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'processed_{dataset_type}.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"💾 تم حفظ البيانات في: {output_path}")
        
        return df