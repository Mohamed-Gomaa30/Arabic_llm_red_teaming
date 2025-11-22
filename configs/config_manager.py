# configs/config_manager.py
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def get_model_config(self) -> Dict[str, Any]:
        # , model_name: str = None
        """Get configuration for specific model"""

        # if model_name and model_name in self.config['models']:
        #     return self.config['models'][model_name]
        return self.config['models']
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config['data']
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get('evaluation', {})
    
    def get_safety_config(self) -> Dict[str, Any]:
        """Get safety evaluation configuration"""
        return self.config.get('safety_evaluation', {})