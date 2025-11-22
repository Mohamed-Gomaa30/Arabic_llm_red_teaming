# """
# ROBOSTAI - Arabic LLM Security Evaluation
# """
# import modal
# from pathlib import Path 
# import os

# configs_dir = Path(__file__) / "configs"
# config_data = Path(__file__) / "data/raw"
# congig_env = Path (__file__) / ".env"
# config_src = Path(__file__) / "src"


# # Modal Image 
# image = (modal.Image.debian_slim(python_version="3.10")
#     .pip_install(
#     "google-generativeai>=0.3.0",
#     "pandas>=1.3.0", 
#     "torch>=2.0.0",
#     "transformers>=4.30.0",
#     "accelerate>=0.20.0",
#     "huggingface_hub>=0.16.0",
#     "bitsandbytes>=0.40.0",
#     "PyYAML>=6.0")

#     .add_local_file(congig_env, remote_path= "/root/")
#     .add_local_dir(configs_dir, remote_path= "/root/config")
#     .add_local_dir(config_src, remote_path="/root/src")
#     .add_local_dir(config_data, remote_path="/root/data/raw")
# )
# app = modal.App("robostai-arabic-llm")

# @app.function(
#     image=image,
#     # gpu="A10G",
#     timeout=1800,
#     secrets=[modal.Secret.from_dotenv(Path(__file__))]
# )
# def evaluate_on_modal(*arglist):
#     """
#     Args: 
#         model: model_name 
#         dataset: dataset_name

#     output: 
#         evalute models 
#     """
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str)
#     parser.add_argument("--dataset", type=str)
#     args = parser.parse_args(args=arglist)

#     dataset_choice = args.model
#     model_choice = args.dataset

#     print("=" * 60)
#     print("ROBOSTAI - Arabic LLM Security Evaluation")
#     print("=" * 60)
#     print(f"Dataset: {dataset_choice}")
#     print(f"Model: {model_choice}")
#     # print(f"Data sample : {sample_size}")
#     print("=" * 60)
    
#     try:

#         from configs.config_manager import ConfigManager
#         from src.data_processor import DataProcessor
#         from src.safety_evaluator import SafetyEvaluator

#         config = ConfigManager("root/config/config.yaml")
        
#         processor = DataProcessor(config)
#         data = processor.process_data(dataset_choice)
        
#         evaluator = SafetyEvaluator(config)
#         results = evaluator.evaluate_model(model_choice, data)
        
#         summary = evaluator.calculate_success_rates(results, model_choice)
        
#         output_dir = config.get_data_config()['output_dir']
#         results_path = os.path.join(output_dir, f'results_{model_choice}_{dataset_choice}.csv')
#         results.to_csv(results_path, index=False, encoding='utf-8')
        
#         print(f"Save Results into:  {results_path}")
        
#         return summary
        
#     except Exception as e:
#         print(f"Error during evaluation {e}")
#         return {"error": str(e)}



# from pathlib import Path 
# import yaml
# config_path = Path(__file__).parent / "configs/config.yaml"


# with open(config_path, 'r') as f:
#     config = yaml.safe_load(f)

# print(config)




from src.multi_format_tester import MultiFormatTester


print("done!")