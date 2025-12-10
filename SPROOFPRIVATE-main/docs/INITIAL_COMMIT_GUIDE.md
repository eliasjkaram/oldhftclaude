# Initial Git Commit Structure Guide for Alpaca-MCP

This guide helps you organize the initial commits for the Alpaca-MCP trading system repository.

## 🎯 Goal

Transform 250+ files in a flat structure into a well-organized, secure Git repository with meaningful commit history.

## 📋 Pre-Commit Checklist

Before ANY commits:

1. ✅ Run `python security_audit.py` - MUST PASS
2. ✅ Review `.gitignore` is working: `git status --ignored`
3. ✅ Ensure `.env` is NOT staged
4. ✅ Set repository to **PRIVATE** on GitHub
5. ✅ Remove all real API keys and credentials

## 🏗️ Recommended Commit Structure

### Phase 1: Foundation (Commits 1-5)

#### Commit 1: Initial Security Setup
```bash
git add .gitignore .env.example security_audit.py pre_push_validate.sh
git add PRE_PUSH_CHECKLIST.md LICENSE_RECOMMENDATIONS.md
git commit -m "security: add security infrastructure and audit tools

- Add comprehensive .gitignore for trading system
- Create security audit script for credential detection  
- Add pre-push validation script
- Include security checklist and guidelines
- Set up .env.example with safe placeholders"
```

#### Commit 2: Documentation and Guidelines
```bash
git add README.md CONTRIBUTING.md LICENSE
git add INITIAL_COMMIT_GUIDE.md
git commit -m "docs: add core documentation and contribution guidelines

- Update README with security warnings and setup instructions
- Add comprehensive CONTRIBUTING.md with coding standards
- Include MIT LICENSE with copyright
- Add initial commit structure guide"
```

#### Commit 3: Development Tools
```bash
git add .pre-commit-config.yaml
git add requirements.txt requirements-dev.txt
git add pyproject.toml setup.py pytest.ini
git commit -m "chore: add development tools and configurations

- Configure pre-commit hooks for security scanning
- Add requirements files for dependencies
- Set up project configuration files
- Include test configuration"
```

#### Commit 4: Core Source Structure
```bash
# Create directory structure first
mkdir -p src/{algorithms,ai,data,execution,risk,utils}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p config/{trading,risk,system}
mkdir -p docs/{api,guides,architecture}

# Add __init__.py files
touch src/__init__.py
touch src/algorithms/__init__.py
# ... (add all __init__.py files)

git add src/ tests/ config/ docs/
git commit -m "feat: establish core project structure

- Create src/ directory with logical modules
- Set up test directory structure  
- Add configuration directories
- Initialize Python packages"
```

#### Commit 5: Configuration Templates
```bash
# Move/create configuration examples
git add config/trading/paper_trading.example.json
git add config/risk/risk_limits.example.json
git add config/system/system_config.example.json
git commit -m "feat: add configuration templates

- Add paper trading configuration example
- Include risk management templates
- Add system configuration examples
- All with safe default values"
```

### Phase 2: Core Components (Commits 6-15)

#### Commit 6-8: Base Classes and Utilities
```bash
# Move core utility files
git add src/utils/config.py src/utils/logging.py
git add src/utils/helpers.py src/utils/decorators.py
git commit -m "feat: add core utilities and helpers

- Configuration management system
- Logging setup with security filtering
- Common helper functions
- Useful decorators for trading system"
```

#### Commit 9-11: Trading Algorithms (Carefully Selected)
```bash
# Add ONLY well-tested, secure algorithms
git add src/algorithms/base.py
git add src/algorithms/arbitrage/simple_arbitrage.py
git add src/algorithms/options/iron_condor.py
git commit -m "feat: add core trading algorithms

- Base algorithm abstract class
- Simple arbitrage detection
- Iron condor options strategy
- All with paper trading safeguards"
```

#### Commit 12-15: Data and Execution
```bash
# Add data management
git add src/data/collectors/base_collector.py
git add src/data/processors/data_cleaner.py
git commit -m "feat: add data collection and processing

- Base data collector interface
- Data cleaning and validation
- Time series alignment utilities"
```

### Phase 3: Testing and Examples (Commits 16-20)

#### Commit 16-18: Test Suite
```bash
git add tests/unit/test_algorithms.py
git add tests/unit/test_risk_management.py
git add tests/fixtures/sample_data.py
git commit -m "test: add comprehensive test suite

- Unit tests for core algorithms
- Risk management test cases
- Test fixtures with synthetic data
- No real trading data included"
```

#### Commit 19-20: Safe Examples
```bash
git add examples/paper_trading_demo.py
git add examples/backtest_example.py
git commit -m "docs: add safe example implementations

- Paper trading demonstration
- Backtesting example with sample data
- Clear warnings about live trading
- Educational purposes only"
```

### Phase 4: Monitoring and Deployment (Commits 21-25)

#### Commit 21-23: Monitoring Setup
```bash
git add monitoring/prometheus.yml
git add monitoring/grafana/dashboards/
git add monitoring/alerts/
git commit -m "feat: add monitoring configuration

- Prometheus configuration for metrics
- Grafana dashboards for visualization
- Alert rules for risk thresholds
- No sensitive endpoints exposed"
```

#### Commit 24-25: Deployment Files
```bash
git add Dockerfile docker-compose.yml
git add deployment/kubernetes/
git add .dockerignore
git commit -m "chore: add containerization and deployment

- Dockerfile with security best practices
- Docker-compose for local development
- Kubernetes manifests (optional)
- Proper .dockerignore for security"
```

### Phase 5: Cleanup (Commits 26-30)

#### Final Commits: Organization
```bash
# After moving files to proper locations
git add -A
git status  # CAREFUL REVIEW
git commit -m "refactor: complete project reorganization

- Move all files to appropriate directories
- Remove duplicates and outdated code
- Ensure no credentials in any files
- Final security audit passed"
```

## 🚫 What NOT to Commit

### Never Commit These Files

1. **Credentials**:
   - `.env` (only `.env.example`)
   - Any file with real API keys
   - Files with passwords or secrets

2. **Trading Data**:
   - Real trading logs
   - Actual position data
   - Personal account information
   - P&L reports with real data

3. **Large Files**:
   - MinIO data files
   - Database files
   - Log files
   - Trained model files (>100MB)

4. **Sensitive Code**:
   - Files with hardcoded credentials
   - Production server details
   - Internal network information

### Handle Problem Files

For files with mixed content (some good code + credentials):

```bash
# Option 1: Clean the file first
python remove_credentials.py problematic_file.py

# Option 2: Add to .gitignore temporarily
echo "problematic_file.py" >> .gitignore

# Option 3: Create sanitized version
cp problematic_file.py problematic_file_sanitized.py
# Edit to remove sensitive data
git add problematic_file_sanitized.py
```

## 📊 Commit Size Guidelines

- **Small commits**: 1-10 files of related functionality
- **Medium commits**: 10-50 files (only for reorganization)
- **Never**: 100+ files in one commit

## 🔍 Verification Steps

After EACH commit:

```bash
# 1. Check what was committed
git show --stat

# 2. Search for credentials
git show | grep -i "api_key\|secret\|password"

# 3. Verify file sizes
git ls-tree -r -l HEAD | awk '{print $4, $5}' | sort -n

# 4. Run security audit on committed files
python security_audit.py
```

## 🚀 Initial Push

When ready to push:

```bash
# 1. Final security check
./pre_push_validate.sh

# 2. Review entire history
git log --oneline --graph --all

# 3. Check for secrets in history
git secrets --scan-history  # If git-secrets installed

# 4. Create private repository on GitHub
# Go to GitHub.com → New Repository → PRIVATE

# 5. Add remote and push
git remote add origin git@github.com:yourusername/alpaca-mcp.git
git push -u origin main

# 6. Immediately check GitHub
# Verify repository is PRIVATE
# Review files through GitHub UI
```

## 🆘 If You Make a Mistake

### Committed a Secret?

```bash
# If not pushed yet:
git reset --soft HEAD~1
# Remove the sensitive file
# Recommit without it

# If already pushed (EMERGENCY):
# 1. Immediately rotate the exposed credential
# 2. Use BFG Repo-Cleaner or git filter-branch
# 3. Force push (coordinate with team)
# 4. Consider the credential compromised
```

### Committed Large Files?

```bash
# Remove from history
git filter-branch --tree-filter 'rm -f path/to/large/file' HEAD
# Or use BFG: bfg --delete-files largefile.bin
```

## 📈 Success Metrics

Your initial commit series is successful when:

- ✅ All commits pass security audit
- ✅ No credentials in repository
- ✅ Clear, logical commit history
- ✅ Files organized in proper structure
- ✅ All tests passing
- ✅ Documentation complete
- ✅ Repository is PRIVATE
- ✅ Team can clone and run paper trading

## 🎉 Final Steps

1. **Tag initial release**: `git tag -a v0.1.0 -m "Initial secure release"`
2. **Create development branch**: `git checkout -b develop`
3. **Set up CI/CD** with security scanning
4. **Document** the setup process for team
5. **Celebrate** - but stay vigilant!

Remember: Security is an ongoing process, not a one-time setup. Stay alert! 🔒