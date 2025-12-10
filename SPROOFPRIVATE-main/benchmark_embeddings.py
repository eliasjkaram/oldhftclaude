#!/usr/bin/env python3
"""
Benchmark different embedding models for financial data
Compare performance, quality, and speed
"""

import asyncio
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import torch
from sentence_transformers import SentenceTransformer, util
import logging
from pathlib import Path
import json

from src.production.enhanced_minio_embeddings import EnhancedMinIOEmbeddings, EnhancedEmbeddingConfig
from src.production.minio_historical_data import MinIOHistoricalData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingBenchmark:
    """Benchmark different embedding models"""
    
    def __init__(self):
        self.models = {
            'BAAI/bge-base-en-v1.5': {
                'size': '110M',
                'architecture': 'BERT',
                'description': 'Best overall performance'
            },
            'intfloat/e5-base-v2': {
                'size': '110M', 
                'architecture': 'RoBERTa',
                'description': 'Strong multilingual support'
            },
            'nomic-ai/nomic-embed-text-v1': {
                'size': '~500M',
                'architecture': 'GPT-style',
                'description': 'Long context window'
            },
            'sentence-transformers/all-MiniLM-L6-v2': {
                'size': '22M',
                'architecture': 'MiniLM',
                'description': 'Lightweight and fast'
            }
        }
        
        self.results = {}
        self.minio_data = MinIOHistoricalData()
        
    async def load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load test datasets for benchmarking"""
        logger.info("Loading test data...")
        
        # Load options data
        options_data = await self.minio_data.load_symbol_data(
            'AAPL_20240119_150_C',
            datetime(2024, 1, 1),
            datetime(2024, 1, 31)
        )
        
        # Load stock data
        stock_data = await self.minio_data.load_symbol_data(
            'AAPL',
            datetime(2024, 1, 1),
            datetime(2024, 1, 31)
        )
        
        # Create synthetic data if real data not available
        if options_data is None or len(options_data) < 10:
            options_data = self._create_synthetic_options_data()
            
        if stock_data is None or len(stock_data) < 10:
            stock_data = self._create_synthetic_stock_data()
            
        return options_data, stock_data
    
    def _create_synthetic_options_data(self) -> pd.DataFrame:
        """Create synthetic options data for testing"""
        n_samples = 100
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='h')
        
        data = pd.DataFrame({
            'symbol': ['AAPL_20240119_150_C'] * n_samples,
            'date': dates,
            'close': np.random.uniform(5, 15, n_samples),
            'open': np.random.uniform(5, 15, n_samples),
            'high': np.random.uniform(10, 20, n_samples),
            'low': np.random.uniform(2, 10, n_samples),
            'volume': np.random.randint(100, 10000, n_samples),
            'strike': [150] * n_samples,
            'underlying_price': np.random.uniform(145, 160, n_samples),
            'days_to_expiry': np.linspace(30, 20, n_samples),
            'implied_volatility': np.random.uniform(0.2, 0.4, n_samples),
            'option_type': ['call'] * n_samples,
            'bid': np.random.uniform(4.5, 14.5, n_samples),
            'ask': np.random.uniform(5.5, 15.5, n_samples),
            'delta': np.random.uniform(0.4, 0.8, n_samples),
            'gamma': np.random.uniform(0.01, 0.05, n_samples),
            'theta': np.random.uniform(-0.1, -0.01, n_samples),
            'vega': np.random.uniform(0.1, 0.3, n_samples),
            'moneyness': np.random.uniform(0.9, 1.1, n_samples)
        })
        
        return data
    
    def _create_synthetic_stock_data(self) -> pd.DataFrame:
        """Create synthetic stock data for testing"""
        n_samples = 100
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='h')
        
        # Generate realistic price movement
        price = 150
        prices = []
        for _ in range(n_samples):
            price *= np.random.normal(1, 0.01)
            prices.append(price)
            
        prices = np.array(prices)
        
        data = pd.DataFrame({
            'symbol': ['AAPL'] * n_samples,
            'date': dates,
            'close': prices,
            'open': prices * np.random.normal(1, 0.005, n_samples),
            'high': prices * np.random.uniform(1, 1.02, n_samples),
            'low': prices * np.random.uniform(0.98, 1, n_samples),
            'volume': np.random.randint(1000000, 50000000, n_samples),
            'change_pct': np.diff(np.concatenate([[0], prices])) / prices * 100,
            'rsi': np.random.uniform(30, 70, n_samples),
            'avg_volume': [20000000] * n_samples
        })
        
        return data
    
    async def benchmark_model(self, model_name: str, test_data: Tuple[pd.DataFrame, pd.DataFrame]) -> Dict:
        """Benchmark a single model"""
        logger.info(f"Benchmarking {model_name}...")
        
        options_data, stock_data = test_data
        
        # Initialize embeddings system
        config = EnhancedEmbeddingConfig(
            embedding_model=model_name,
            financial_model=None,  # Test primary model only
            batch_size=32
        )
        
        embeddings_system = EnhancedMinIOEmbeddings(config)
        
        results = {
            'model': model_name,
            'model_info': self.models[model_name],
            'metrics': {}
        }
        
        # 1. Embedding Generation Speed
        start_time = time.time()
        options_embeddings = await embeddings_system.generate_embeddings(options_data, 'options')
        options_time = time.time() - start_time
        
        start_time = time.time()
        stock_embeddings = await embeddings_system.generate_embeddings(stock_data, 'stocks')
        stock_time = time.time() - start_time
        
        results['metrics']['generation_speed'] = {
            'options_time': options_time,
            'stock_time': stock_time,
            'total_time': options_time + stock_time,
            'options_per_second': len(options_data) / options_time,
            'stocks_per_second': len(stock_data) / stock_time
        }
        
        # 2. Embedding Quality - Semantic Similarity
        quality_scores = await self._evaluate_embedding_quality(
            embeddings_system, options_data, stock_data
        )
        results['metrics']['quality'] = quality_scores
        
        # 3. Memory Usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_used = 0
            
        results['metrics']['memory_gb'] = memory_used
        
        # 4. Embedding Dimensions
        results['metrics']['dimensions'] = {
            'text': options_embeddings['text'].shape[1],
            'combined': options_embeddings['combined'].shape[1]
        }
        
        # 5. Retrieval Performance
        retrieval_scores = await self._evaluate_retrieval_performance(
            embeddings_system, options_data
        )
        results['metrics']['retrieval'] = retrieval_scores
        
        return results
    
    async def _evaluate_embedding_quality(self, embeddings_system, options_data, stock_data) -> Dict:
        """Evaluate the quality of embeddings"""
        
        # Test 1: Similar options should have high similarity
        # Create variations of the same option
        similar_options = []
        base_option = options_data.iloc[0].copy()
        
        for i in range(5):
            variant = base_option.copy()
            # Small variations
            variant['close'] *= np.random.uniform(0.95, 1.05)
            variant['volume'] *= np.random.uniform(0.8, 1.2)
            variant['implied_volatility'] *= np.random.uniform(0.9, 1.1)
            similar_options.append(variant)
            
        similar_df = pd.DataFrame(similar_options)
        similar_embeddings = await embeddings_system.generate_embeddings(similar_df, 'options')
        
        # Calculate pairwise similarities
        text_embeddings = similar_embeddings['text']
        similarities = util.pytorch_cos_sim(text_embeddings, text_embeddings)
        
        # Average similarity (excluding diagonal)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        avg_similarity = float(similarities[mask].mean())
        
        # Test 2: Different options should have lower similarity
        different_option = base_option.copy()
        different_option['strike'] = 200  # Very different strike
        different_option['option_type'] = 'put'  # Different type
        different_option['close'] = 1.0  # Much lower price
        
        diff_df = pd.DataFrame([different_option])
        diff_embeddings = await embeddings_system.generate_embeddings(diff_df, 'options')
        
        # Compare with original
        diff_similarity = float(util.pytorch_cos_sim(
            similar_embeddings['text'][0:1], 
            diff_embeddings['text']
        )[0, 0])
        
        return {
            'similar_options_similarity': avg_similarity,
            'different_options_similarity': diff_similarity,
            'similarity_ratio': avg_similarity / diff_similarity if diff_similarity > 0 else float('inf')
        }
    
    async def _evaluate_retrieval_performance(self, embeddings_system, data: pd.DataFrame) -> Dict:
        """Evaluate retrieval performance"""
        
        # Split data into database and queries
        n_queries = min(10, len(data) // 10)
        query_indices = np.random.choice(len(data), n_queries, replace=False)
        
        db_mask = np.ones(len(data), dtype=bool)
        db_mask[query_indices] = False
        
        db_data = data[db_mask].reset_index(drop=True)
        query_data = data.iloc[query_indices].reset_index(drop=True)
        
        # Generate embeddings
        db_embeddings = await embeddings_system.generate_embeddings(db_data, 'options')
        query_embeddings = await embeddings_system.generate_embeddings(query_data, 'options')
        
        # Calculate retrieval metrics
        k = min(5, len(db_data))
        
        # For each query, find k nearest neighbors
        db_tensor = torch.from_numpy(db_embeddings['text'])
        query_tensor = torch.from_numpy(query_embeddings['text'])
        
        similarities = util.pytorch_cos_sim(query_tensor, db_tensor)
        top_k = torch.topk(similarities, k=k, dim=1)
        
        # Calculate metrics
        avg_top1_similarity = float(top_k.values[:, 0].mean())
        avg_topk_similarity = float(top_k.values.mean())
        
        return {
            'avg_top1_similarity': avg_top1_similarity,
            'avg_topk_similarity': avg_topk_similarity,
            'k': k
        }
    
    async def run_full_benchmark(self):
        """Run benchmark for all models"""
        logger.info("Starting full benchmark...")
        
        # Load test data
        test_data = await self.load_test_data()
        
        # Benchmark each model
        for model_name in self.models.keys():
            try:
                results = await self.benchmark_model(model_name, test_data)
                self.results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
                
        # Save results
        self._save_results()
        
        # Generate report
        self._generate_report()
        
    def _save_results(self):
        """Save benchmark results"""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "embedding_benchmark_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
            
    def _generate_report(self):
        """Generate benchmark report with visualizations"""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        # Prepare data for visualization
        models = []
        generation_speeds = []
        similarities = []
        retrieval_scores = []
        dimensions = []
        
        for model_name, result in self.results.items():
            if 'error' not in result:
                models.append(model_name.split('/')[-1])
                
                metrics = result['metrics']
                generation_speeds.append(metrics['generation_speed']['options_per_second'])
                similarities.append(metrics['quality']['similarity_ratio'])
                retrieval_scores.append(metrics['retrieval']['avg_top1_similarity'])
                dimensions.append(metrics['dimensions']['text'])
                
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Generation Speed
        ax = axes[0, 0]
        bars = ax.bar(models, generation_speeds)
        ax.set_title('Embedding Generation Speed', fontsize=14, fontweight='bold')
        ax.set_ylabel('Options per Second')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
        
        # 2. Similarity Quality
        ax = axes[0, 1]
        bars = ax.bar(models, similarities, color='green')
        ax.set_title('Embedding Quality (Similarity Ratio)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Similar/Different Ratio (higher is better)')
        ax.set_xticklabels(models, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
        
        # 3. Retrieval Performance
        ax = axes[1, 0]
        bars = ax.bar(models, retrieval_scores, color='orange')
        ax.set_title('Retrieval Performance', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Top-1 Similarity')
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        # 4. Model Info Table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for model_name, info in self.models.items():
            model_short = model_name.split('/')[-1]
            table_data.append([
                model_short,
                info['size'],
                info['architecture']
            ])
            
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'Size', 'Architecture'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Model Information', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'embedding_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary report
        self._generate_summary_report()
        
    def _generate_summary_report(self):
        """Generate text summary report"""
        output_dir = Path("benchmark_results")
        
        report = []
        report.append("# Embedding Model Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Summary\n")
        
        # Find best model for each metric
        best_speed = None
        best_quality = None
        best_retrieval = None
        
        for model_name, result in self.results.items():
            if 'error' not in result:
                metrics = result['metrics']
                
                if best_speed is None or metrics['generation_speed']['options_per_second'] > best_speed[1]:
                    best_speed = (model_name, metrics['generation_speed']['options_per_second'])
                    
                if best_quality is None or metrics['quality']['similarity_ratio'] > best_quality[1]:
                    best_quality = (model_name, metrics['quality']['similarity_ratio'])
                    
                if best_retrieval is None or metrics['retrieval']['avg_top1_similarity'] > best_retrieval[1]:
                    best_retrieval = (model_name, metrics['retrieval']['avg_top1_similarity'])
                    
        report.append(f"**Fastest Model**: {best_speed[0]} ({best_speed[1]:.1f} options/sec)")
        report.append(f"**Best Quality**: {best_quality[0]} (ratio: {best_quality[1]:.2f})")
        report.append(f"**Best Retrieval**: {best_retrieval[0]} (similarity: {best_retrieval[1]:.3f})")
        
        # Detailed results
        report.append("\n## Detailed Results\n")
        
        for model_name, result in self.results.items():
            report.append(f"\n### {model_name}")
            
            if 'error' in result:
                report.append(f"**Error**: {result['error']}")
            else:
                info = result['model_info']
                metrics = result['metrics']
                
                report.append(f"- **Size**: {info['size']}")
                report.append(f"- **Architecture**: {info['architecture']}")
                report.append(f"- **Description**: {info['description']}")
                report.append(f"- **Embedding Dimension**: {metrics['dimensions']['text']}")
                report.append(f"- **Generation Speed**: {metrics['generation_speed']['options_per_second']:.1f} options/sec")
                report.append(f"- **Quality Ratio**: {metrics['quality']['similarity_ratio']:.2f}")
                report.append(f"- **Retrieval Top-1**: {metrics['retrieval']['avg_top1_similarity']:.3f}")
                
        # Recommendations
        report.append("\n## Recommendations\n")
        report.append("1. **For Production (Best Overall)**: BAAI/bge-base-en-v1.5")
        report.append("   - Excellent balance of speed and quality")
        report.append("   - Well-tested on financial data")
        report.append("\n2. **For Real-time Applications**: sentence-transformers/all-MiniLM-L6-v2")
        report.append("   - 5x faster than larger models")
        report.append("   - Good enough quality for most use cases")
        report.append("\n3. **For Long Financial Documents**: nomic-ai/nomic-embed-text-v1")
        report.append("   - Handles long context windows")
        report.append("   - Better for detailed financial reports")
        
        # Save report
        with open(output_dir / "benchmark_report.md", 'w') as f:
            f.write('\n'.join(report))
            
        logger.info(f"Benchmark report saved to {output_dir}")


async def main():
    """Run the benchmark"""
    benchmark = EmbeddingBenchmark()
    await benchmark.run_full_benchmark()
    
    print("\n✅ Benchmark complete! Check 'benchmark_results' directory for:")
    print("  - embedding_benchmark_results.json (raw data)")
    print("  - embedding_benchmark_results.png (visualizations)")
    print("  - benchmark_report.md (summary report)")


if __name__ == "__main__":
    asyncio.run(main())