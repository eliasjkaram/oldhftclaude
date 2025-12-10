# 🚨 PRE-PUSH SECURITY CHECKLIST 🚨

**DO NOT PUSH TO GIT UNTIL ALL ITEMS ARE CHECKED!**

## 🔒 Critical Security Checks

- [ ] **Run Security Audit**
  ```bash
  python security_audit.py
  ```
  - Must return 0 (no critical issues)
  - Review all findings in `security_audit_results.json`

- [ ] **Check for .env files**
  ```bash
  ls -la | grep "\.env"
  ```
  - Only `.env.example` and `.env.template` should exist
  - NO `.env`, `.env.local`, `.env.production` files

- [ ] **Verify API Keys are removed**
  ```bash
  grep -r "sk-or-v1-" . --include="*.py" --include="*.json"
  grep -r "PKAP" . --include="*.py" --include="*.json"
  ```
  - Should return NO results

- [ ] **Check CLAUDE.md**
  - Remove or replace the OpenRouter API key mentioned
  - This file contains: `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`

## 📁 Repository Structure

- [ ] **Create repository as PRIVATE first**
  - GitHub: Settings → Danger Zone → Change visibility (only after cleanup)
  
- [ ] **Verify .gitignore is comprehensive**
  - Check that all sensitive patterns are included
  - Test with: `git status --ignored`

## 🗑️ Data and Storage

- [ ] **Remove large data files**
  ```bash
  find . -size +10M -type f
  ```
  - Move to external storage or add to .gitignore

- [ ] **Clean up logs**
  ```bash
  rm -f *.log
  rm -rf logs/
  ```

- [ ] **Remove database files**
  ```bash
  rm -f *.db *.sqlite *.db-journal
  ```

- [ ] **Clean MinIO data**
  ```bash
  rm -rf minio-data/ minio_cache/
  rm -f minio_inventory*.json
  ```

## 🧹 Code Cleanup

- [ ] **Remove hardcoded IPs/URLs**
  ```bash
  grep -r "127\.0\.0\.1\|localhost" . --include="*.py" | grep -v example
  ```

- [ ] **Check for test credentials**
  ```bash
  grep -r "test.*password\|dummy.*key" . --include="*.py" -i
  ```

- [ ] **Remove debug prints with sensitive data**
  ```bash
  grep -r "print.*api.*key\|print.*secret" . --include="*.py" -i
  ```

## 📋 Final Verification

- [ ] **Dry run git add**
  ```bash
  git add --dry-run .
  ```
  - Review all files that would be added

- [ ] **Check git status**
  ```bash
  git status
  ```
  - Ensure no sensitive files in staging

- [ ] **Review first commit**
  ```bash
  git diff --cached
  ```
  - Final review before committing

## 🚀 Safe Push Process

1. **Initialize repository**
   ```bash
   git init
   git add .gitignore .env.example README.md
   git commit -m "Initial commit with safe files"
   ```

2. **Add source code gradually**
   ```bash
   git add src/
   git status  # Review
   git commit -m "Add source code"
   ```

3. **Add documentation**
   ```bash
   git add docs/ *.md
   git commit -m "Add documentation"
   ```

4. **Final security check**
   ```bash
   python security_audit.py
   ```

5. **Push to PRIVATE repository**
   ```bash
   git remote add origin git@github.com:YOUR_USERNAME/alpaca-mcp-private.git
   git push -u origin main
   ```

## ⚠️ If Credentials Were Exposed

If you accidentally committed credentials:

1. **Immediately revoke all exposed credentials**
   - Alpaca: https://app.alpaca.markets/
   - OpenRouter: Regenerate API keys
   - Change all passwords

2. **Remove from Git history**
   ```bash
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch PATH_TO_FILE" \
     --prune-empty --tag-name-filter cat -- --all
   ```

3. **Force push** (coordinate with team)
   ```bash
   git push --force --all
   git push --force --tags
   ```

4. **Consider the credentials permanently compromised**
   - Even after removal, assume they were captured
   - Monitor for suspicious activity

## 📝 Pre-commit Hook

Install the pre-commit hook to automate checks:

```bash
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "🔒 Running security audit..."
python security_audit.py
if [ $? -ne 0 ]; then
    echo "❌ Security audit failed! Fix issues before committing."
    exit 1
fi
echo "✅ Security audit passed!"
EOF

chmod +x .git/hooks/pre-commit
```

---

**Remember**: It's better to be overly cautious with financial trading systems. When in doubt, don't commit it!