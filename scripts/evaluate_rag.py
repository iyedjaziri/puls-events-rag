"""
RAG Evaluation Module
Evaluate RAG system performance using RAGAS metrics with Mistral LLM.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from scripts.rag_system import EventRAGSystem

# Load environment variables
load_dotenv()

class RAGEvaluator:
    """Evaluate RAG system performance using RAGAS and Mistral."""
    
    def __init__(self, rag_system: EventRAGSystem):
        """
        Initialize evaluator.
        
        Args:
            rag_system: Initialized RAG system
        """
        self.rag_system = rag_system
        
        # Initialize Mistral LLM for RAGAS (as judge)
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
            
        self.eval_llm = ChatMistralAI(
            mistral_api_key=api_key,
            model="mistral-small-latest",
            temperature=0
        )
        
        # Use local embeddings for evaluation to save API calls/cost
        # and match the system's embedding space
        self.eval_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
    def evaluate(
        self,
        test_dataset_path: str,
        output_path: str = "results/evaluation_results.json"
    ) -> Dict:
        """
        Evaluate RAG system on test dataset using RAGAS.
        
        Args:
            test_dataset_path: Path to annotated test dataset JSON
            output_path: Path to save evaluation results
            
        Returns:
            Dictionary with evaluation metrics and analysis
        """
        # Load test dataset
        logger.info(f"Loading test dataset from {test_dataset_path}")
        if not os.path.exists(test_dataset_path):
            raise FileNotFoundError(f"Test dataset not found at {test_dataset_path}")
            
        with open(test_dataset_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        logger.info(f"Loaded {len(test_data)} test cases")
        
        # Prepare data for RAGAS
        questions = []
        ground_truths = []
        answers = []
        contexts = []
        
        results = []
        
        # 1. Run Inference
        logger.info("Running inference on test dataset...")
        for i, test_case in enumerate(test_data):
            logger.info(f"Processing test case {i+1}/{len(test_data)}")
            
            question = test_case["question"]
            reference_answer = test_case.get("reference_answer", "")
            
            try:
                # Get RAG response
                start_time = time.time()
                response = self.rag_system.ask(question, k=5)
                latency = (time.time() - start_time) * 1000
                
                predicted_answer = response["answer"]
                # Extract context texts
                retrieved_contexts = [s["text"] for s in response.get("sources", [])]
                
                # Store for RAGAS
                questions.append(question)
                ground_truths.append([reference_answer])
                answers.append(predicted_answer)
                contexts.append(retrieved_contexts)
                
                # Store individual result
                results.append({
                    "test_id": i,
                    "question": question,
                    "predicted_answer": predicted_answer,
                    "reference_answer": reference_answer,
                    "latency_ms": latency,
                    "sources": response.get("sources", [])[:3]
                })
                
            except Exception as e:
                logger.error(f"Error processing test case {i}: {e}")
                continue

        if not results:
            logger.error("No results generated. Aborting evaluation.")
            return {}

        # 2. Run RAGAS Evaluation
        logger.info("Running RAGAS evaluation (this may take a while)...")
        
        # Create dataset
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        try:
            ragas_results = ragas_evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                llm=self.eval_llm,
                embeddings=self.eval_embeddings
            )
            
            final_metrics = ragas_results
            logger.info("RAGAS evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            final_metrics = {"error": str(e)}
        
        # 3. Save Results
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        evaluation_report = {
            "summary": final_metrics,
            "detailed_results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "llm_model": "mistral-small-latest",
                "embedding_model": "paraphrase-multilingual-mpnet-base-v2"
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Evaluation results saved to {output_path}")
        
        return evaluation_report


def main():
    """Example usage."""
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    try:
        rag = EventRAGSystem(
            faiss_index_path="faiss_index/events.index",
            metadata_path="faiss_index/metadata.pkl",
            config_path="faiss_index/config.json",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            top_k=5
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return
    
    # Initialize evaluator
    try:
        evaluator = RAGEvaluator(rag)
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return
    
    # Run evaluation
    results = evaluator.evaluate(
        test_dataset_path="data/test/test_dataset_annotated.json",
        output_path="results/evaluation_results.json"
    )
    
    # Print summary
    if "summary" in results:
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        summary = results["summary"]
        if "error" in summary:
            print(f"Evaluation Error: {summary['error']}")
        else:
            print(f"\nRAGAS Metrics:")
            for metric, value in summary.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        print(f"\n✓ Full results saved to results/evaluation_results.json")
    else:
        print("\nEvaluation failed to generate results.")


if __name__ == "__main__":
    main()
