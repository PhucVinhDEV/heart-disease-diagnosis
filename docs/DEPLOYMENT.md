# Deployment Guide

## Live Application

**URL:** https://heart-disease-diagnosis-vietai.streamlit.app

**Status:** Production (24/7 availability)

---

## Streamlit Cloud Deployment

### Prerequisites

1. GitHub account
2. Streamlit Cloud account (free tier)
3. Repository pushed to GitHub

### Deployment Steps

```bash
# 1. Ensure code is pushed
git add .
git commit -m "Ready for deployment"
git push origin main

# 2. Deploy on Streamlit Cloud
# - Visit https://share.streamlit.io/
# - Click "New app"
# - Select repository: huynqhcmus/heart-disease-diagnosis
# - Branch: main
# - Main file: app/streamlit_app.py
# - Click "Deploy"

# 3. Wait 3-5 minutes for initial deployment
```

### Configuration Files

**Required files in repository:**
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit settings
- `packages.txt` - System packages (currently empty)
- `.gitignore` - Git exclusions

---

## Auto-Deployment

Changes pushed to `main` branch trigger automatic redeployment:

```bash
git add modified_files
git commit -m "Update feature X"
git push

# Streamlit Cloud automatically:
# 1. Detects changes
# 2. Pulls latest code
# 3. Reinstalls dependencies
# 4. Restarts application
# Time: ~2-3 minutes
```

---

## Resource Limits (Free Tier)

- **CPU:** 1 vCPU
- **RAM:** 1 GB
- **Storage:** Limited
- **Bandwidth:** Unlimited
- **Apps:** 1 private + unlimited public

**Optimization strategies:**
- Model caching with `@st.cache_resource`
- Lazy loading for large files
- Compressed model files

---

## Monitoring

### Application Logs

View real-time logs in Streamlit Cloud dashboard:
1. Go to https://share.streamlit.io/
2. Select app
3. Click "Manage app" → "Logs"

### Health Check

```bash
curl https://heart-disease-diagnosis-vietailearningteam.streamlit.app/healthz
```

---

## Troubleshooting

### App Won't Start

**Check:**
1. Logs for error messages
2. All dependencies in `requirements.txt`
3. Python version compatibility (3.10+)

**Solution:**
```bash
# Update requirements
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push
```

### Out of Memory

**Symptoms:**
- App crashes under load
- "Memory limit exceeded" error

**Solutions:**
1. Reduce number of loaded models
2. Implement lazy loading
3. Clear cache periodically

### Slow Performance

**Optimize:**
```python
# Cache expensive operations
@st.cache_resource
def load_models():
    # Model loading code
    return models

@st.cache_data
def load_data():
    # Data loading code
    return df
```

---

## Environment Variables

Set secrets in Streamlit Cloud dashboard:

```toml
# Settings → Secrets
# Add any API keys or sensitive data
# Example:
# API_KEY = "your-key-here"
```

Access in code:
```python
import streamlit as st
api_key = st.secrets["API_KEY"]
```

---

## Rollback

Revert to previous version:

```bash
# Method 1: Git revert
git revert HEAD
git push

# Method 2: Redeploy specific commit
# In Streamlit Cloud dashboard:
# Settings → Advanced → Reboot app → Select commit
```

---

## Custom Domain (Optional)

Streamlit Cloud Pro allows custom domains:

**Setup:**
1. Upgrade to Pro plan
2. Dashboard → Settings → Custom domain
3. Add CNAME record in DNS: `app.yourdomain.com → share.streamlit.io`

---

## Local Development

Test before deployment:

```bash
# Run locally
./scripts/run_app.sh

# Access at http://localhost:8501

# Test all features:
# - Model loading
# - Predictions
# - Report generation
# - History management
```

---

## Best Practices

1. ✓ Test locally before pushing
2. ✓ Use descriptive commit messages
3. ✓ Monitor logs after deployment
4. ✓ Implement error handling
5. ✓ Cache expensive operations
6. ✓ Keep dependencies minimal

---

## Support

**Issues:** https://github.com/huynqhcmus/heart-disease-diagnosis/issues  
**Streamlit Docs:** https://docs.streamlit.io/  
**Community:** https://discuss.streamlit.io/

---

*Last updated: September 30, 2025*
