import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_content = file.read()
            
            # استبدال environment variables
            config_content = os.path.expandvars(config_content)
            
            return yaml.safe_load(config_content)
        except FileNotFoundError:
            raise Exception(f"Config file not found: {self.config_path}")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        return self.config['models'].get(model_name, {})
    
    def get_data_config(self) -> Dict[str, Any]:
        return self.config['data']
    
    def get_modal_config(self) -> Dict[str, Any]:
        return self.config.get('modal', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        return self.config['evaluation']
    
    def should_use_modal(self) -> bool:
        return self.get_modal_config().get('enable_modal', False)