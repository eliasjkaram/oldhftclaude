# Complete MinIO Embeddings Generation & Query System

## 🚀 Overview

This system generates and stores embeddings for **ALL** MinIO financial data (2002-2025) using 4 state-of-the-art embedding models, enabling powerful similarity search, pattern recognition, and financial analysis.

### Coverage
- **Time Period**: 2002-2025 (24 years)
- **Data Types**: Stocks and Options
- **Volume**: ~250,000+ options contracts per day + major stocks
- **Models**: 4 best-in-class embedding models
- **Storage**: MinIO cloud storage with FAISS indices

## 📦 System Components

### 1. **Embedding Generation** (`generate_all_embeddings.py`)
- Processes all data year by year
- Generates embeddings using 4 models simultaneously
- Automatic checkpointing and resume capability
- Handles billions of data points efficiently

### 2. **Progress Monitoring** (`monitor_embedding_generation.py`)
- Real-time progress dashboard
- Visual progress tracking
- Performance statistics
- Report generation

### 3. **Query System** (`query_stored_embeddings.py`)
- Natural language search across all embeddings
- Multi-model comparison
- Pattern evolution analysis
- Arbitrage opportunity detection

### 4. **Control Script** (`start_embedding_generation.sh`)
- Easy-to-use menu system
- Process management
- Automatic error recovery

## 🎯 Quick Start

### Step 1: Install Dependencies

```bash
pip install sentence-transformers faiss-gpu torch minio pandas numpy matplotlib rich tqdm
```

### Step 2: Start Generation

```bash
./start_embedding_generation.sh
```

Choose option 1 to start new generation.

### Step 3: Monitor Progress

In another terminal:
```bash
python monitor_embedding_generation.py
```

Or use the menu option 3 for live monitoring.

## 🔧 Models Used

| Model | Size | Best For | Dimension |
|-------|------|----------|-----------|
| **BAAI/bge-base-en-v1.5** | 110M | Best overall quality | 768 |
| **intfloat/e5-base-v2** | 110M | Multi-language support | 768 |
| **nomic-ai/nomic-embed-text-v1** | 500M | Long documents | 768 |
| **sentence-transformers/all-MiniLM-L6-v2** | 22M | Real-time queries | 384 |

## 📊 Storage Structure

```
MinIO Bucket: embeddings/
└── financial-embeddings/
    ├── year_2002/
    │   ├── batch_0.pkl
    │   ├── batch_1.pkl
    │   └── ...
    ├── year_2003/
    │   └── ...
    ├── indices/
    │   ├── bge-base_year_2002.faiss
    │   ├── bge-base_year_2003.faiss
    │   ├── e5-base_year_2002.faiss
    │   └── ...
    ├── checkpoints/
    │   ├── year_2002_complete.json
    │   └── ...
    └── generation_summary.json
```

## 🔍 Query Examples

### 1. Natural Language Search

```python
from query_stored_embeddings import StoredEmbeddingsQuery

query_system = StoredEmbeddingsQuery()

# Search across all years and models
results = await query_system.search_text(
    "Deep in the money Apple calls with high volume before earnings",
    k=20
)

for result in results:
    print(f"{result.symbol} ({result.year}): {result.similarity:.3f}")
```

### 2. Specific Model/Year Search

```python
# Use fastest model for recent years only
results = await query_system.search_text(
    "SPY puts with unusual options activity",
    k=10,
    models=['minilm'],  # Fast model
    years=[2023, 2024]  # Recent years only
)
```

### 3. Pattern Evolution

```python
# Analyze how volatility patterns changed over time
evolution = await query_system.analyze_pattern_evolution(
    "High implied volatility before market crash",
    start_year=2008,
    end_year=2024
)

for year, matches in evolution.items():
    print(f"{year}: {len(matches)} similar patterns found")
```

### 4. Find Similar Options

```python
# Find historically similar options
similar = await query_system.search_similar_options(
    reference_data=current_option_df,
    k=50,
    model='bge-base',  # Best quality
    years=[2020, 2021, 2022, 2023, 2024]
)
```

## ⚡ Performance Optimization

### Generation Speed
- **GPU**: ~1000 embeddings/second
- **CPU**: ~100 embeddings/second
- **Total Time**: ~48-72 hours for full dataset on GPU

### Query Speed
- **Text Search**: <100ms for single query
- **Batch Search**: Process 1000 queries in <10 seconds
- **Index Loading**: <1 second per year/model

### Memory Usage
- **Generation**: 16-32GB RAM recommended
- **Query**: 4-8GB RAM sufficient
- **GPU Memory**: 8GB+ recommended

## 🛠️ Advanced Configuration

### Custom Generation Config

```python
from generate_all_embeddings import EmbeddingGenerationConfig

config = EmbeddingGenerationConfig(
    start_year=2020,  # Process subset
    end_year=2024,
    batch_size=2000,  # Larger batches for GPU
    max_workers=16,   # More parallel workers
    use_gpu=True,
    checkpoint_frequency=50  # More frequent saves
)
```

### Selective Model Generation

To generate embeddings for specific models only:

```python
config.models = {
    'bge-base': 'BAAI/bge-base-en-v1.5',  # Best quality only
    # Comment out others
}
```

## 📈 Use Cases

### 1. Historical Pattern Matching
Find options that behaved similarly to current opportunities

### 2. Arbitrage Detection
Identify pricing discrepancies between similar options

### 3. Risk Analysis
Find positions with similar risk profiles across history

### 4. Market Regime Detection
Identify when similar market conditions occurred

### 5. Strategy Backtesting
Find historical scenarios matching current conditions

### 6. Options Flow Analysis
Detect unusual options activity patterns

## 🔧 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python generate_all_embeddings.py --batch-size 500
```

### GPU Not Available
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python generate_all_embeddings.py
```

### Resume After Failure
The system automatically resumes from the last checkpoint:
```bash
./start_embedding_generation.sh
# Choose option 2: Resume
```

### Verify Data Integrity
```python
# Check completed years
python monitor_embedding_generation.py report
```

## 📊 Statistics

After full generation:
- **Total Embeddings**: ~2 billion vectors
- **Storage Size**: ~500GB compressed
- **Indices**: 96 (4 models × 24 years)
- **Query Coverage**: 100% of historical data

## 🚨 Important Notes

1. **First Run**: Takes 48-72 hours on GPU
2. **Storage**: Requires ~500GB in MinIO
3. **Network**: Stable connection required
4. **Resume**: System handles interruptions gracefully
5. **Updates**: Can incrementally add new data

## 📝 Maintenance

### Add New Year's Data
```python
# Just run for the new year
config = EmbeddingGenerationConfig(
    start_year=2026,
    end_year=2026
)
```

### Rebuild Specific Index
```python
# Rebuild if corrupted
await generator.build_faiss_indices({
    'year': 2024,
    'embeddings': year_data
})
```

### Clean Old Checkpoints
```bash
# Remove old checkpoints (keep indices!)
mc rm --recursive minio/embeddings/financial-embeddings/checkpoints/
```

## 🎯 Best Practices

1. **Start Small**: Test with 1 year first
2. **Monitor Progress**: Keep monitoring dashboard open
3. **Use Checkpoints**: System saves progress frequently
4. **Query Testing**: Test queries before full generation
5. **Model Selection**: 
   - Use `bge-base` for best quality
   - Use `minilm` for speed
   - Use `nomic` for long descriptions

## 🔗 Next Steps

After generation completes:

1. **Build Applications**: Use query system in trading apps
2. **Create Alerts**: Monitor for similar patterns
3. **Backtest Strategies**: Find historical analogs
4. **Research**: Analyze pattern evolution
5. **Optimize**: Fine-tune models on your specific use case

---

For support or questions, check the individual component documentation or run with `--help` flag.