"""
ROBOSTAI - Arabic LLM Security Evaluation
"""
import modal
from pathlib import Path 
import os
current_dir = Path(__file__).parent
configs_dir = current_dir / "configs"
config_data = current_dir / "data/raw"
congig_env = current_dir / ".env"
config_src = current_dir / "src"

volume_name = "robostai-arabic-llm-data"
output_volume = modal.Volume.from_name(volume_name, create_if_missing=True)
print(f"Volume out {output_volume}")
# Modal Image 


image = (modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "python-dotenv", 
            "google-generativeai", 
            "langchain-huggingface==0.2.0",
            "pandas>=1.3.0", 
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate==1.7.0",
            "huggingface_hub>=0.16.0",
            "bitsandbytes>=0.40.0",
            "mishkal==0.4.1",
            "PyYAML>=6.0", 
            "google-generativeai>=0.3.0",
            "einops", 
            "safetensors",
            "sentencepiece", 
            "tiktoken",
            "autoawq"
        )

    # local dir, or files to remote execution env 
    .add_local_file(congig_env, remote_path= "/root/")
    .add_local_dir(configs_dir, remote_path= "/root/configs/")
    .add_local_dir(config_src, remote_path="/root/src/")
    .add_local_dir(config_data, remote_path="/root/data/raw/")
)
app = modal.App("robostai-arabic-llm")
@app.function(
    image=image,
    gpu="A100-40GB",# for 7B models 
    # gpu="A100-80GB", # for 13B models  
    timeout=1800,
    volumes={"/root/processed/":output_volume}, 
    secrets=[modal.Secret.from_dotenv(Path(__file__))]
)
def evaluate(*arglist):
    """
    Args: 
        model: model_name 
        dataset: dataset_name
    """

    import sys
    sys.path.insert(0, '/root/')
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['harmful', 'regional'], required=True)
    parser.add_argument("--models", type=str, nargs='+', default=['gemini', 'jais', 'acegpt', 'allam'])
    parser.add_argument("--sample_size", type=int, default=5)
    args = parser.parse_args(args=arglist)
    
    print("=" * 60)
    print("ROBOSTAI - Arabic LLM Security Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print("=" * 60)
    
    try:
        from configs.config_manager import ConfigManager
        from src.multi_format_tester import MultiFormatTester

        # Initialize
        config = ConfigManager("/root/configs/config.yaml")
        tester = MultiFormatTester(config)
        
        # Test datasets
        all_results = {}
        
        if args.dataset == 'harmful':
            print("Testing harmful dataset...")
            harmful_results = tester.test_all_formats(
                args.models, 'harmful', args.sample_size
            )
            all_results['harmful'] = harmful_results
        
        if args.dataset == 'regional':
            print("Testing regional dataset...")
            regional_results = tester.test_all_formats(
                args.models, 'regional', args.sample_size
            )
            all_results['regional'] = regional_results
        
        print(f"\n{'='*60}")
        print("TESTING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results saved to: /root/processed/")
        print(f"Check the 'results' and 'summary_reports' directories")
        
        return {
            "status": "success",
            "datasets_tested": list(all_results.keys()),
            "total_prompts": sum(len(results['results']) for results in all_results.values() if 'results' in results)
        }
    
    except Exception as e:
        import traceback
        print(f"Error during evaluation: {e}")
        print(traceback.format_exc()) 
        return {"error": str(e)}
