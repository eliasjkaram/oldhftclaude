# MinIO Multi-Vector Embeddings System

A comprehensive embeddings system for analyzing 23+ years (2002-2025) of stocks and options data stored in MinIO.

## Overview

This system creates multi-modal embeddings from MinIO datasets, enabling:
- Similarity search across millions of options contracts
- Pattern recognition and historical analysis
- Arbitrage opportunity detection
- Market regime change detection
- Advanced querying and visualization

## Architecture

### Multi-Vector Embedding Types

1. **Price Embeddings (128d)**
   - Price trajectories and movements
   - Volatility patterns
   - Volume-weighted features
   - Momentum indicators

2. **Greeks Embeddings (64d)**
   - Delta, Gamma, Theta, Vega, Rho
   - Higher-order Greeks (Lambda, Vanna)
   - Greeks dynamics over time

3. **Microstructure Embeddings (64d)**
   - Bid-ask spreads
   - Order flow imbalance
   - Volume patterns
   - Price impact metrics

4. **Technical Embeddings (128d)**
   - Moving averages (5, 10, 20, 50, 100, 200 day)
   - RSI, MACD, Bollinger Bands
   - ATR, Stochastic Oscillator
   - 50+ technical indicators

5. **Fused Embeddings (256d)**
   - Cross-modal attention fusion
   - Unified representation
   - Optimized for similarity search

### Data Coverage

- **2002-2009**: Historical options in `/aws_backup/extracted_options/`
- **2010-2019**: Options data in `/options-complete/`
- **2020-2025**: Recent data in `/options-recent/` and `/options/`
- **~250,000+ options contracts per day**

## Installation

```bash
# Install required packages
pip install torch faiss-cpu numpy pandas matplotlib seaborn streamlit plotly scikit-learn

# For GPU support
pip install faiss-gpu torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Generate Embeddings

```bash
# Generate embeddings for entire dataset (2002-2025)
python run_minio_embeddings.py generate --full-index

# Generate embeddings year by year with testing
python run_minio_embeddings.py generate --start-year 2020 --end-year 2025 --test-after-year

# Generate for specific years
python run_minio_embeddings.py generate --start-year 2023 --end-year 2024
```

### 2. Query Embeddings

```bash
# Find similar options
python run_minio_embeddings.py query AAPL_20240119_150_C 2024-01-15 -k 20

# Test embeddings for a specific year
python run_minio_embeddings.py test --year 2024
```

### 3. Interactive Interface

```bash
# Launch Streamlit interface
streamlit run src/production/embeddings_query_interface.py
```

## Python API

```python
from src.production.comprehensive_minio_embeddings import ComprehensiveMinIOEmbeddings, EmbeddingConfig
from src.production.embeddings_query_interface import EmbeddingsQueryInterface
import asyncio
from datetime import datetime, timedelta

# Initialize system
config = EmbeddingConfig(
    start_year=2002,
    end_year=2025,
    batch_size=1024,
    max_workers=16
)
embeddings_system = ComprehensiveMinIOEmbeddings(config)

# Generate embeddings for a period
async def generate_embeddings():
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    results = await embeddings_system.generate_embeddings_for_period(
        start_date, end_date, 
        symbols=['AAPL', 'SPY', 'AAPL_20240119_150_C']
    )
    return results

# Search for similar patterns
async def search_patterns():
    # Load query data
    query_data = await embeddings_system.minio_data.load_symbol_data(
        'AAPL_20240119_150_C',
        datetime(2024, 1, 1),
        datetime(2024, 1, 15)
    )
    
    # Search
    results = await embeddings_system.search_similar_patterns(
        query_data,
        k=20,
        embed_types=['fused', 'price', 'greeks']
    )
    return results

# Find arbitrage opportunities
async def find_arbitrage():
    interface = EmbeddingsQueryInterface(embeddings_system)
    
    opportunities = await interface.find_arbitrage_opportunities(
        current_date=datetime(2024, 1, 15),
        min_spread=0.02,
        max_risk=0.1
    )
    return opportunities

# Run examples
asyncio.run(generate_embeddings())
asyncio.run(search_patterns())
asyncio.run(find_arbitrage())
```

## Query Examples

### 1. Find Similar Options
```python
# Find options similar to AAPL call
similar = await interface.find_similar_options(
    'AAPL_20240119_150_C',
    (datetime(2024, 1, 1), datetime(2024, 1, 31)),
    similarity_threshold=0.8,
    max_results=50
)
```

### 2. Detect Market Regime Changes
```python
# Detect regime changes in 2020-2024
changes = await interface.find_regime_changes(
    datetime(2020, 1, 1),
    datetime(2024, 1, 1),
    sensitivity=2.0
)
```

### 3. Analyze Historical Patterns
```python
# Analyze volatility smile patterns
analysis = await interface.analyze_historical_patterns(
    'volatility_smile',
    start_year=2002,
    end_year=2025
)
```

## Performance Optimization

### GPU Acceleration
- Automatically uses GPU if available
- Batch processing for efficiency
- Parallel data loading

### Caching
- Local cache in `/tmp/embeddings_cache/`
- MinIO-based persistent storage
- Checkpoint saving every 5 years

### Index Types
- FAISS IVF1024,PQ64 for large-scale search
- Proximity graphs for advanced retrieval
- Multiple index types for different use cases

## Advanced Features

### 1. Multi-Modal Fusion
- Cross-attention between different embedding types
- Weighted combination based on query type
- Adaptive fusion based on data characteristics

### 2. Temporal Analysis
- Time-aware embeddings
- Sequence modeling with LSTM/Transformer
- Historical pattern evolution tracking

### 3. Market Microstructure
- Order flow analysis
- Bid-ask spread dynamics
- Volume profile embeddings

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce batch_size in EmbeddingConfig
2. **Slow Generation**: Increase max_workers for parallel processing
3. **Missing Data**: Check MinIO credentials and data paths

### Performance Tips

1. Use GPU for faster embedding generation
2. Generate embeddings in chunks for large datasets
3. Use appropriate index types for your use case
4. Enable caching for repeated queries

## Architecture Details

### Embedding Model
```
MultiModalEmbeddingModel
├── Price Encoder (Bidirectional LSTM)
├── Greeks Encoder (MLP)
├── Microstructure Encoder (Transformer)
├── Technical Encoder (Deep MLP)
└── Cross-Modal Fusion Layer
```

### Storage Structure
```
MinIO Bucket
└── comprehensive_vectors/
    ├── indices/
    │   ├── price_index.faiss
    │   ├── greeks_index.faiss
    │   ├── microstructure_index.faiss
    │   ├── technical_index.faiss
    │   └── fused_index.faiss
    ├── checkpoints/
    │   ├── checkpoint_2005.pkl
    │   ├── checkpoint_2010.pkl
    │   └── ...
    └── metadata.json
```

## Future Enhancements

1. **Real-time Updates**: Continuous embedding updates
2. **Advanced Queries**: Natural language querying
3. **Graph Embeddings**: Options chain relationships
4. **Cross-Asset Analysis**: Correlation embeddings
5. **AutoML Integration**: Automated feature engineering

## Support

For issues or questions:
- Check existing implementations in `src/production/`
- Review test cases in query interface
- Examine MinIO data structure documentation