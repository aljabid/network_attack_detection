#  Quick Start Guide

Get your AI-Based Network Attack Detection System running in **5 minutes**!

---

## Step 1: Extract the Project

```bash

# Unzip the project (if downloaded as zip)
unzip network_attack_detection.zip
cd network_attack_detection
```

## Step 2: Run Setup

```bash

# Make setup script executable
chmod +x setup.sh

# Run setup (installs all dependencies)
./setup.sh
```

This will:
- ✅ Create virtual environment
- ✅ Install all Python packages
- ✅ Create necessary directories
- ✅ Verify installation

## Step 3: Train Models

```bash

# Activate virtual environment
source venv/bin/activate

# Train models (uses sample data if datasets not available)
./run.sh train
```

**Expected output:**
```
Training Random Forest...
Random Forest - Accuracy: 0.9650
Random Forest - F1 Score: 0.9500

Training XGBoost...
XGBoost - Accuracy: 0.9720
...

✓ Training complete!
Models saved to: data/models/
```

## Step 4: Test Real-Time Detection

```bash

# Start real-time detection service
./run.sh detect
```

**Expected output:**
```
  Attack detected: ddos (confidence: 95.3%)
  Attack detected: port_scan (confidence: 87.2%)
...
```

---

## Alternative: Use Docker

If you prefer Docker:

```bash

# Build image
docker build -t attack-detector .

# Run container
docker run -d \
  --name detector \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  attack-detector
```

Or use Docker Compose:

```bash

docker-compose up -d
```

---

## Verify Installation

```bash

# Run tests
./run.sh test

# Check visualizations
ls -lh dashboards/

# View logs
tail -f logs/system.log
```

---

## Quick Commands Reference

| Command | Description |
|---------|-------------|
| `./run.sh train` | Train all models |
| `./run.sh detect` | Start real-time detection |
| `./run.sh test` | Run test suite |
| `./run.sh notebook` | Launch Jupyter notebook |
| `./run.sh visualize` | Generate visualizations |
| `./run.sh clean` | Clean temporary files |
| `./run.sh help` | Show all commands |

---

## Next Steps

1. **Review Documentation**: Read `README.md` for complete information
2. **Explore Notebook**: Run `./run.sh notebook` to see interactive tutorial
3. **Deploy to Production**: Follow `DEPLOYMENT.md` for production setup
4. **Customize**: Edit `configs/config.yaml` to adjust settings

---

## Troubleshooting

### "Command not found"
```bash

chmod +x run.sh setup.sh
```

### "Module not found"
```bash

source venv/bin/activate
pip install -r requirements.txt
```

### "Permission denied"
```bash

sudo chown -R $USER:$USER .
chmod -R 755 .
```

---

## Support

-  Full documentation: `README.md`
-  Deployment guide: `DEPLOYMENT.md`
-  Project overview: `PROJECT_OVERVIEW.md`
-  Run tests: `./run.sh test`

---

That's it! Your system is ready to detect network attacks! 
