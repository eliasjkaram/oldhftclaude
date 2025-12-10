# MinIO Embeddings Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
pip install sentence-transformers faiss-cpu torch numpy pandas
```

### 2. Quick Test

```python
from src.production.enhanced_minio_embeddings import EnhancedMinIOEmbeddings
import asyncio
import pandas as pd

async def quick_test():
    # Initialize system
    embeddings = EnhancedMinIOEmbeddings()
    
    # Create sample options data
    sample_data = pd.DataFrame({
        'symbol': ['AAPL_20240119_150_C'],
        'close': [10.5],
        'strike': [150],
        'implied_volatility': [0.25],
        'option_type': ['call']
    })
    
    # Generate embeddings
    results = await embeddings.generate_embeddings(sample_data, 'options')
    print(f"Generated embeddings: {list(results.keys())}")
    
    # Search similar
    similar = await embeddings.search_similar(
        "Find Apple call options near the money",
        k=5
    )
    print(f"Found {len(similar['results'])} similar options")

asyncio.run(quick_test())
```

### 3. Generate Embeddings for Your Data

```bash
# For full dataset (takes time)
python run_minio_embeddings.py generate --full-index

# For specific years
python run_minio_embeddings.py generate --start-year 2023 --end-year 2024

# For testing
python run_minio_embeddings.py test --year 2024
```

### 4. Query Embeddings

```bash
# Find similar options
python run_minio_embeddings.py query AAPL_20240119_150_C 2024-01-15 -k 20

# Launch interactive interface
streamlit run src/production/embeddings_query_interface.py
```

## 📊 Model Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| **BAAI/bge-base-en-v1.5** | 110M | Fast | Excellent | Production |
| **sentence-transformers/all-MiniLM-L6-v2** | 22M | Very Fast | Good | Real-time |
| **nomic-ai/nomic-embed-text-v1** | 500M | Moderate | Excellent | Long docs |
| **intfloat/e5-base-v2** | 110M | Fast | Very Good | Multi-language |

## 🎯 Common Use Cases

### 1. Find Similar Options
```python
# Find options similar to a current position
results = await embeddings.search_similar(
    current_option_data,
    k=20,
    search_type='combined'
)
```

### 2. Natural Language Search
```python
# Search using plain English
results = await embeddings.search_similar(
    "Deep in the money Apple calls with high volume",
    k=10
)
```

### 3. Arbitrage Detection
```python
from src.production.embeddings_query_interface import EmbeddingsQueryInterface

interface = EmbeddingsQueryInterface()
opportunities = await interface.find_arbitrage_opportunities(
    current_date=datetime.now(),
    min_spread=0.02
)
```

### 4. Pattern Recognition
```python
# Analyze historical patterns
analysis = await interface.analyze_historical_patterns(
    'volatility_smile',
    start_year=2020,
    end_year=2024
)
```

## 🔧 Configuration Options

### High Quality (Slower)
```python
config = EnhancedEmbeddingConfig(
    embedding_model="BAAI/bge-base-en-v1.5",
    financial_model="nomic-ai/nomic-embed-text-v1",
    batch_size=256,
    use_gpu=True
)
```

### Fast (Real-time)
```python
config = EnhancedEmbeddingConfig(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    batch_size=1024,
    use_gpu=True
)
```

### Balanced
```python
config = EnhancedEmbeddingConfig(
    embedding_model="intfloat/e5-base-v2",
    batch_size=512
)
```

## 📈 Performance Tips

1. **Use GPU**: 10x faster embedding generation
2. **Batch Processing**: Process multiple queries together
3. **Cache Results**: Embeddings are cached automatically
4. **Index Selection**: Use IVF indices for large datasets

## 🛠️ Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Use lightweight model for large datasets
- Enable CPU offloading

### Slow Search
- Increase `nprobe` for better accuracy
- Use GPU-enabled FAISS
- Pre-filter data before embedding

### Poor Quality Results
- Use BAAI/bge-base model
- Ensure data preprocessing is correct
- Increase `k` for more results

## 📚 Examples

Run practical examples:
```bash
# Run all examples
python embeddings_usage_examples.py

# Run specific example (1-10)
python embeddings_usage_examples.py 3
```

Example topics:
1. Find similar trades
2. Volatility regime detection
3. Arbitrage scanner
4. Pattern recognition
5. Earnings play analysis
6. Risk similarity search
7. Market sentiment
8. Pairs trading
9. Options flow analysis
10. Backtesting scenarios

## 🔗 Resources

- [Full Documentation](MINIO_EMBEDDINGS_README.md)
- [Benchmark Results](benchmark_embeddings.py)
- [API Reference](src/production/enhanced_minio_embeddings.py)

## 💡 Pro Tips

1. **Combine Embedding Types**: Use both text and numerical embeddings for best results
2. **Time-Aware Queries**: Include date ranges in your searches
3. **Risk-Based Search**: Search by Greeks and risk metrics
4. **Pattern Templates**: Save successful queries as templates

## 🚨 Quick Wins

1. **Similar Options Finder**: Instantly find historical options similar to current opportunities
2. **Arbitrage Scanner**: Real-time detection of price discrepancies
3. **Pattern Matcher**: Identify complex multi-leg strategies
4. **Sentiment Analysis**: Gauge market sentiment from options flow

Start with example #1 to see immediate results!