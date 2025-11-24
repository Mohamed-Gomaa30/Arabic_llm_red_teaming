# src/llm_executor.py
import os
import time
import logging
from typing import List, Dict
import google.generativeai as genai
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from configs.config_manager import ConfigManager
from dotenv import load_dotenv

dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path)

HF_Token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
Gemini_Token = os.environ.get("GEMINI_API_KEY")


logger = logging.getLogger(__name__)

class LLMExecutor:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_config = config.get_model_config()
        
        self.models = {}
        self.tokenizers = {}
        self._setup_models()

    def _setup_models(self):
        """Initialize models - load HuggingFace models directly"""
        try:
            if 'gemini' in self.model_config:
                if Gemini_Token:
                    genai.configure(api_key=Gemini_Token)
                    safety_settings = [
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE"
                        }
                    ]
                    self.models['gemini'] = genai.GenerativeModel("gemini-2.5-flash-lite", safety_settings=safety_settings)
                    logger.info("Gemini client initialized")
                else:
                    logger.warning("GEMINI_API_KEY not found")
            
            logger.info("HuggingFace models will be loaded on first use")
                    
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise
    
    def _load_hf_model(self, model_name: str):
        """Load HuggingFace model"""
        model_path = self.model_config[model_name]['model_name']
        
        if model_name in self.models:
            return          
        try:
            logger.info(f"Loading {model_name} from {model_path}...")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                model_path,
                token=HF_Token, 
                trust_remote_code=True
            )
            logger.info("Tokenizer Loaded ...")

            if self.tokenizers[model_name].pad_token is None:
                self.tokenizers[model_name].pad_token = self.tokenizers[model_name].eos_token

            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_path,
                token=HF_Token, 
                dtype=torch.float32, # if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                revision="main" # Explicitly set revision to avoid warning if possible, or just ignore
            )
            self.models[model_name].eval()
            logger.info("Model Loaded ...")

            logger.info(f"{model_name} loaded successfully on device: {self.models[model_name].device}")
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
        
    def query_gemini(self, prompt: str) -> str:
        """Query Gemini model"""
        try:
            gemini_config = self.model_config.get('gemini', {})
            temperature = gemini_config.get('temperature', 0.0)
            max_tokens = gemini_config.get('max_tokens', 1000)
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]

            response = self.models['gemini'].generate_content(
                [
                    {"role": "user", "parts": [prompt]}
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                ),
                safety_settings=safety_settings
            )

            try:
                return response.text
            except ValueError:
                finish_reason = None
                if hasattr(response, "candidates") and response.candidates:
                    finish_reason = getattr(response.candidates[0], "finish_reason", None)
                logger.warning(f"Gemini API: No text returned. finish_reason={finish_reason}")
                return f"ERROR: Gemini did not return text. finish_reason={finish_reason}"
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"ERROR: {str(e)}"
    
    def query_jais(self, prompt: str) -> str:
        """Query JAIS model"""
        try:
            self._load_hf_model('jais')
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
            
            inputs = self.tokenizers['jais'](formatted_prompt, return_tensors="pt").input_ids
            inputs = input_ids.to(device)
            # inputs = {k: v.to(self.models['jais'].device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.models['jais'].generate(
                    inputs,
                    max_new_tokens=1000,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizers['jais'].eos_token_id
                )
            
            response = self.tokenizers['jais'].decode(outputs[0], skip_special_tokens=True)
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"JAIS inference error: {e}")
            return f"ERROR: {str(e)}"
    
    def query_acegpt(self, prompt: str) -> str:
        """Query AceGPT model"""
        try:
            self._load_hf_model('acegpt')
            
            formatted_prompt = f"Human: {prompt}\n\nAssistant:"
            
            inputs = self.tokenizers['acegpt'](formatted_prompt, return_tensors="pt")
            inputs = {k: v.to(self.models['acegpt'].device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.models['acegpt'].generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.0,
                    do_sample=False,
                    eos_token_id=self.tokenizers['acegpt'].eos_token_id,
                    pad_token_id=self.tokenizers['acegpt'].eos_token_id
                )
            
            response = self.tokenizers['acegpt'].decode(outputs[0], skip_special_tokens=True)
            if "Assistant:" in response:
                response = response.split("Assistant:")[1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"AceGPT inference error: {e}")
            return f"ERROR: {str(e)}"
    
    def query_allam(self, prompt: str) -> str:
        """Query Allam model"""
        try:
            self._load_hf_model('allam')

            formatted_prompt = f"### Human: {prompt}\n### Assistant:"
            
            inputs = self.tokenizers['allam'](formatted_prompt, return_tensors="pt")
            inputs = {k: v.to(self.models['allam'].device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.models['allam'].generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.0,
                    do_sample=False,
                    eos_token_id=self.tokenizers['allam'].eos_token_id,
                    pad_token_id=self.tokenizers['allam'].eos_token_id
                )
            
            response = self.tokenizers['allam'].decode(outputs[0], skip_special_tokens=True)
            if "### Assistant:" in response:
                response = response.split("### Assistant:")[1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Allam inference error: {e}")
            return f"ERROR: {str(e)}"
    
    def query_model(self, model: str, prompt: str) -> str:
        """Query specified model"""
        if model == 'gemini':
            return self.query_gemini(prompt)
        elif model == 'jais':
            # print("---------------------------------------------------")
            return self.query_jais(prompt)
        elif model == 'acegpt':
            return self.query_acegpt(prompt)
        elif model == 'allam':
            return self.query_allam(prompt)
        else:
            return f"ERROR: Model {model} not supported"
    
    def run_batch_test(self, df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
        """Run batch testing on all models for all text types"""
        results = []
        
        # Text representations to test
        text_columns = {
            'arabic_original': 'Arabic',
            'arabizi_no_numbers': 'arabizi_no_numbers',
            'arabizi_with_numbers': 'arabizi_with_numbers',
            'transliteration': 'transliteration',
            'diacritized': 'diacritized'
        }
        
        for model in models:
            if model not in ['gemini', 'jais', 'acegpt', 'allam']:
                logger.warning(f"Model {model} not supported, skipping")
                continue
                
            logger.info(f"Testing model: {model}")
            
            for text_type, column_name in text_columns.items():
                if column_name not in df.columns:
                    logger.warning(f"Column {column_name} not found in dataframe, skipping")
                    continue
                    
                logger.info(f"  Testing text type: {text_type}")
                
                for idx, row in df.iterrows():
                    prompt = str(row[column_name])
                    if not prompt.strip():
                        continue
                    
                    logger.info(f"    Prompt {idx+1}: {prompt[:50]}...")
                    response = self.query_model(model, prompt)
                    logger.info(f"    Response: {response[:100]}...")
                    
                    result = {
                        'id': row['Id'],
                        'taxonomy_level1': row.get('Taxnomy| Level 1', ''),
                        'taxonomy_level2': row.get('Taxnomy| Level 2', ''),
                        'taxonomy_level3': row.get('Taxnomy| Level 3', ''),
                        'q_type': row.get('Q-Type', ''),
                        'model': model,
                        'text_type': text_type,
                        'prompt': prompt,
                        'response': response,
                        'prompt_full': prompt
                    }
                    
                    results.append(result)

                    # Time delay to handle the gemini response rate 
                    time.sleep(2)

        
        return pd.DataFrame(results)