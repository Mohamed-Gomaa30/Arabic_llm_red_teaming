# src/multi_format_tester.py (updated)
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
        self.model_name = None
        
    def test_all_formats(self, models: List[str], dataset_name: str, 
                        sample_size: int = None) -> Dict:
        """Test all text formats and prepare for manual evaluation"""
        logger.info(f"Starting multi-format testing for {dataset_name}")

        # Model_name 
        self.model_name = models[0]

        # Load processed data
        processed_path = os.path.join(self.output_dir, f'processed_{dataset_name}.csv')
        
        if os.path.exists(processed_path):
            logger.info(f"Loading preprocessed data from: {processed_path}")
            dataset = pd.read_csv(processed_path)
        else:
            logger.info(f"Processing data for: {dataset_name}")
            dataset = self.data_processor.process_data(dataset_name)
        
        # Sample data if needed
        if sample_size > 1 and sample_size < len(dataset):
            dataset = dataset.sample(sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} prompts from dataset")
        
        logger.info(f"Testing {len(dataset)} prompts on models: {models}")
        
        # Run batch testing
        results_df = self.llm_executor.run_batch_test(dataset, models)
        
        if results_df.empty:
            logger.warning("No results generated")
            return {'results': pd.DataFrame(), 'metrics': {}}
        
        # Create manual annotation template 
        annotation_template = self.safety_evaluator.create_annotation_template(
            results_df, self.output_dir
        )
        
        logger.info("MANUAL EVALUATION REQUIRED")
        print(f"Annotation template created: {annotation_template}")
        logger.info(f"Annotation template created: {annotation_template}")
        logger.info("Please have human annotators categorize responses according to the paper's methodology")
        
        return {
            'results': results_df,
            'annotation_template': annotation_template,
            'message': 'Manual evaluation required - see annotation template'
        }
    
    def analyze_manual_annotations(self, dataset_name: str, annotation_file: str = None):
        """Analyze manually annotated results"""
        if annotation_file is None:
            annotation_file = os.path.join(self.output_dir, f"manual_annotations_{dataset_name}.csv")
        
        if not os.path.exists(annotation_file):
            logger.error(f"Annotation file not found: {annotation_file}")
            return None
        
        # Load manual annotations
        annotated_df = self.safety_evaluator.load_manual_annotations(annotation_file)
        
        # Calculate metrics
        metrics = self.safety_evaluator.calculate_manual_metrics(annotated_df)
        
        # Generate comprehensive report
        self._generate_manual_report(metrics, annotated_df, dataset_name)
        
        return {
            'annotated_results': annotated_df,
            'metrics': metrics
        }
    
    def _generate_manual_report(self, metrics: Dict, annotated_df: pd.DataFrame, dataset: str):
        """Generate comprehensive report from manual annotations"""
        import json
        from datetime import datetime
        
        report_dir = os.path.join(self.output_dir, 'manual_reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # Detailed analysis
        analysis = {
            'dataset': dataset,
            'timestamp': datetime.now().isoformat(),
            'total_responses': len(annotated_df),
            'metrics': metrics,
            'key_insights': self._extract_manual_insights(metrics, annotated_df)
        }
        # Save detailed report
        report_file = os.path.join(report_dir, f"{self.model_name}_{dataset}_manual_analysis.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Print summary to console
        self._print_manual_summary(metrics, dataset)
        
        logger.info(f"Manual analysis report saved to: {report_file}")
    
    def _extract_manual_insights(self, metrics: Dict, annotated_df: pd.DataFrame) -> List[str]:
        """Extract key insights from manual annotations"""
        insights = []
        
        overall = metrics.get('overall', {})
        insights.append(f"Overall jailbreak success rate: {overall.get('jailbreak_success_rate', 0)}%")
        
        # Find most vulnerable model
        model_rates = {}
        for key, value in metrics.items():
            if key.startswith('model_'):
                model_name = key.replace('model_', '')
                model_rates[model_name] = value['jailbreak_success_rate']
        
        if model_rates:
            most_vulnerable = max(model_rates, key=model_rates.get)
            insights.append(f"Most vulnerable model: {most_vulnerable}")
        
        # Category insights
        categories = metrics.get('category_distribution', {})
        if categories:
            most_common_category = max(categories, key=categories.get)
            insights.append(f"Most common response type: {most_common_category}")
        
        return insights
    
    def _print_manual_summary(self, metrics: Dict, dataset: str):
        """Print manual evaluation summary to console"""
        print(f"\n{'='*60}")
        print(f"MANUAL EVALUATION SUMMARY - {dataset.upper()}")
        print(f"{'='*60}")
        
        overall = metrics.get('overall', {})
        print(f"Overall Jailbreak Success Rate: {overall.get('jailbreak_success_rate', 0)}%")
        print(f"Total Responses: {overall.get('total_prompts', 0)}")
        print(f"Successful Jailbreaks: {overall.get('successful_jailbreaks', 0)}")
        
        print(f"\nModel Performance:")
        for key, value in metrics.items():
            if key.startswith('model_'):
                model_name = key.replace('model_', '')
                print(f"  {model_name}: {value['jailbreak_success_rate']}%")
        
        print(f"\nResponse Categories:")
        categories = metrics.get('category_distribution', {})
        for category, count in categories.items():
            print(f"  {category}: {count}")