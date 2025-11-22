# src/multi_format_tester.py
import os
import pandas as pd
import logging
from typing import Dict, List
from src.data_processor import DataProcessor
from src.llm_executor import LLMExecutor
from src.safety_evaluator import SafetyEvaluator
from configs.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class MultiFormatTester:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.llm_executor = LLMExecutor(config)
        self.safety_evaluator = SafetyEvaluator(config)
        self.output_dir = config.get_data_config()['output_dir']
        
    def test_all_formats(self, models: List[str], dataset_name: str, 
                        sample_size: int = None) -> Dict:
        """Test all text formats on specified models"""
        logger.info(f"Starting multi-format testing for {dataset_name}")
        
        # Load processed data (your preprocessing already created this)
        processed_path = os.path.join(self.output_dir, f'processed_{dataset_name}.csv')
        
        if os.path.exists(processed_path):
            logger.info(f"Loading preprocessed data from: {processed_path}")
            dataset = pd.read_csv(processed_path)
        else:
            logger.info(f"Processing data for: {dataset_name}")
            dataset = self.data_processor.process_data(dataset_name)
        
        # Sample data if needed
        if sample_size and sample_size < len(dataset):
            dataset = dataset.sample(sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} prompts from dataset")
        
        logger.info(f"Testing {len(dataset)} prompts on models: {models}")
        
        # Run batch testing
        results_df = self.llm_executor.run_batch_test(dataset, models)
        
        if results_df.empty:
            logger.warning("No results generated")
            return {'results': pd.DataFrame(), 'metrics': {}}
        
        # Evaluate safety
        evaluated_results = self.safety_evaluator.evaluate_responses(results_df)
        metrics = self.safety_evaluator.calculate_jailbreak_metrics(evaluated_results)
        
        # Save results by format
        self._save_results_by_format(evaluated_results, dataset_name)
        
        # Generate summary report
        self._generate_summary_report(metrics, dataset_name)
        
        return {
            'results': evaluated_results,
            'metrics': metrics
        }
    
    def _save_results_by_format(self, results: pd.DataFrame, dataset: str):
        """Save results organized by text format"""
        formats = results['text_type'].unique()
        
        for format_name in formats:
            format_results = results[results['text_type'] == format_name]
            
            # Create format directory
            format_dir = os.path.join(self.output_dir, 'results', format_name)
            os.makedirs(format_dir, exist_ok=True)
            
            # Save by model
            for model in format_results['model'].unique():
                model_results = format_results[format_results['model'] == model]
                filename = os.path.join(format_dir, f"{model}_{dataset}_results.csv")
                model_results.to_csv(filename, index=False, encoding='utf-8')
                logger.info(f"Saved {format_name} results for {model}: {len(model_results)} prompts")
    
    def _generate_summary_report(self, metrics: Dict, dataset: str):
        """Generate summary report"""
        import json
        from datetime import datetime
        
        report_dir = os.path.join(self.output_dir, 'summary_reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report = {
            'dataset': dataset,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'key_findings': self._extract_key_findings(metrics)
        }
        
        filename = os.path.join(report_dir, f"{dataset}_summary.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summary report to: {filename}")
        
        # Also print summary to console
        self._print_console_summary(metrics, dataset)
        
        return report
    
    def _extract_key_findings(self, metrics: Dict) -> List[str]:
        """Extract key findings from metrics"""
        findings = []
        
        if 'format_ranking' in metrics:
            ranking = metrics['format_ranking']
            findings.append(f"Most effective jailbreak format: {ranking['most_effective']}")
            findings.append(f"Least effective jailbreak format: {ranking['least_effective']}")
        
        # Find most vulnerable model
        model_rates = {}
        for key, value in metrics.items():
            if key.startswith('model_'):
                model_name = key.replace('model_', '')
                model_rates[model_name] = value['jailbreak_success_rate']
        
        if model_rates:
            most_vulnerable = max(model_rates, key=model_rates.get)
            most_secure = min(model_rates, key=model_rates.get)
            findings.append(f"Most vulnerable model: {most_vulnerable}")
            findings.append(f"Most secure model: {most_secure}")
        
        overall_rate = metrics.get('overall', {}).get('jailbreak_success_rate', 0)
        findings.append(f"Overall jailbreak success rate: {overall_rate}%")
        
        return findings
    
    def _print_console_summary(self, metrics: Dict, dataset: str):
        """Print summary to console"""
        print(f"\n{'='*60}")
        print(f"SUMMARY FOR {dataset.upper()} DATASET")
        print(f"{'='*60}")
        
        overall = metrics.get('overall', {})
        print(f"Overall Jailbreak Success Rate: {overall.get('jailbreak_success_rate', 0)}%")
        print(f"Total Prompts Tested: {overall.get('total_prompts', 0)}")
        print(f"Successful Jailbreaks: {overall.get('successful_jailbreaks', 0)}")
        
        if 'format_ranking' in metrics:
            print(f"\nFormat Effectiveness Ranking:")
            ranking = metrics['format_ranking']['ranking']
            for format_name, rate in ranking.items():
                print(f"  {format_name}: {rate}%")
        
        print(f"\nModel Performance:")
        for key, value in metrics.items():
            if key.startswith('model_'):
                model_name = key.replace('model_', '')
                print(f"  {model_name}: {value['jailbreak_success_rate']}%")