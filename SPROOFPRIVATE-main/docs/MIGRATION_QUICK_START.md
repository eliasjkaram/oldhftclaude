# 🚀 Migration Quick Start Guide

## Prerequisites
- Python 3.8+
- Git repository initialized
- Backup of current codebase

## Step 1: Run File Analysis (2 minutes)
```bash
# Navigate to migration tools
cd tools/migration

# Run the categorization tool
python categorize_files.py
```

This creates:
- `file_categorization_report.md` - Human-readable report
- `migration_mapping.json` - Automated migration plan
- `file_analysis.json` - Detailed dependency analysis

## Step 2: Review Categorization (5 minutes)
```bash
# Check the categorization report
cat file_categorization_report.md

# Review the proposed file movements
cat migration_mapping.json | head -20
```

## Step 3: Test Migration (Dry Run) (5 minutes)
```bash
# Run a dry run to see what would happen
python migrate_files.py --mapping migration_mapping.json

# Check the dry run report
cat migration_dry_run_*.log
```

## Step 4: Execute Migration (10 minutes)
```bash
# Create a backup first
tar -czf backup_before_migration_$(date +%Y%m%d_%H%M%S).tar.gz *.py *.md *.json

# Execute the actual migration
python migrate_files.py --execute --mapping migration_mapping.json

# Check results
cat migration_report_*.md
```

## Step 5: Verify Migration (5 minutes)
```bash
# Check new structure
tree src/ -d -L 3

# Validate Python imports
python -c "import src.core; import src.data; import src.execution; print('✅ Basic imports working')"

# Run a simple test
python -m src.core.health_monitor
```

## Rollback (If Needed)
```bash
# If something goes wrong, rollback
python migrate_files.py --rollback

# Or restore from backup
tar -xzf backup_before_migration_*.tar.gz
```

## Common Issues & Solutions

### Issue: Import Errors
```bash
# Fix common import issues
find src/ -name "*.py" -exec sed -i 's/^from \([a-zA-Z_]*\) import/from src.\1 import/g' {} \;
```

### Issue: Missing __init__.py
```bash
# Create all missing __init__.py files
find src/ -type d -exec touch {}/__init__.py \;
```

### Issue: Circular Imports
```python
# Check for circular imports
python tools/migration/check_circular_imports.py
```

## Post-Migration Checklist

- [ ] All files moved to correct locations
- [ ] No files left in root directory (except configs)
- [ ] All imports updated and working
- [ ] Tests passing
- [ ] Documentation updated
- [ ] Team notified of changes

## Next Steps

1. Update your IDE/editor project settings
2. Update CI/CD pipelines with new paths
3. Update deployment scripts
4. Train team on new structure
5. Update documentation

## File Structure After Migration

```
alpaca-mcp/
├── src/
│   ├── core/           # Configuration, logging, database
│   ├── data/           # Market data, MinIO, preprocessing  
│   ├── execution/      # Orders, routing, positions
│   ├── strategies/     # Arbitrage, options, HFT
│   ├── ml/            # Models, training, GPU
│   ├── risk/          # Portfolio, VaR, Greeks
│   ├── backtesting/   # Testing engines, validators
│   ├── monitoring/    # Dashboards, alerts, metrics
│   ├── bots/          # Trading bots
│   └── integration/   # External APIs
├── scripts/           # Launchers, maintenance, tools
├── tests/            # Unit, integration, performance
├── configs/          # All configuration files
├── deployment/       # Docker, K8s, scripts
├── docs/            # Documentation
├── examples/        # Demos, tutorials
└── tools/          # Development tools
```

## Questions?

Check the detailed `MIGRATION_PLAN.md` for comprehensive information about the migration process.