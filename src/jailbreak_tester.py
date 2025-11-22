# src/jailbreak_tester.py
import pandas as pd
import logging
from typing import Dict, List
from src.llm_executor import LLMExecutor
from src.safety_evaluator import SafetyEvaluator

logger = logging.getLogger(__name__)

class JailbreakTester:
    def __init__(self):
        self.llm_executor = LLMExecutor()
        self.safety_evaluator = SafetyEvaluator()
        logger.info("JailbreakTester initialized")
    
    def run_test_suite(self, 
                      harmful_df: pd.DataFrame = None,
                      regional_df: pd.DataFrame = None,
                      models: List[str] = None,
                      sample_size: int = 10) -> Dict:
        """Run complete jailbreak test suite"""
        
        if models is None:
            models = ['gemini', 'jais', 'acegpt', 'allam']
        
        all_results = []
        
        # Test harmful dataset
        if harmful_df is not None:
            harmful_sample = harmful_df.sample(min(sample_size, len(harmful_df)))
            logger.info(f"Testing {len(harmful_sample)} harmful prompts")
            harmful_results = self.llm_executor.run_batch_test(harmful_sample, models)
            harmful_results['dataset'] = 'harmful'
            all_results.append(harmful_results)
        
        # Test regional dataset
        if regional_df is not None:
            regional_sample = regional_df.sample(min(sample_size, len(regional_df)))
            logger.info(f"Testing {len(regional_sample)} regional prompts")
            regional_results = self.llm_executor.run_batch_test(regional_sample, models)
            regional_results['dataset'] = 'regional'
            all_results.append(regional_results)
        
        # Combine and evaluate
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            evaluated_results = self.safety_evaluator.evaluate_responses(combined_results)
            metrics = self.safety_evaluator.calculate_metrics(evaluated_results)
            
            return {
                'results': evaluated_results,
                'metrics': metrics
            }
        else:
            return {'results': pd.DataFrame(), 'metrics': {}}