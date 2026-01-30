# Cataract Detection with XAI - Deployment Guide

## ✅ Local Setup (COMPLETED)

Your app is now running locally at: **http://localhost:8501**

### What We Did:
1. ✅ Created virtual environment
2. ✅ Installed all dependencies
3. ✅ Fixed hardcoded paths to use relative paths
4. ✅ Started Streamlit server

### To Run Locally Again:
```bash
cd /Users/rutwik/Documents/Projects/Cataract_Detection_with_XAI
source venv/bin/activate
streamlit run multipage/1_Home.py
```

### To Stop the Server:
Press `Ctrl + C` in the terminal

---

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - FREE)

**Pros:** Free, easy, automatic updates from GitHub
**Cons:** Public by default, limited resources

**Steps:**
1. **Push your code to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Sign in** with your GitHub account

4. **Click "New app"**

5. **Fill in the details:**
   - Repository: `RutwikPatel13/Cataract_Detection_with_XAI`
   - Branch: `main`
   - Main file path: `multipage/1_Home.py`

6. **Click "Deploy"**

7. **Wait 5-10 minutes** for deployment

**Important Notes:**
- The .h5 file (113 MB) needs to be in your GitHub repo or use Git LFS
- Add `.streamlit/secrets.toml` for any sensitive data (not tracked in git)

---

### Option 2: Heroku

**Pros:** More control, can handle larger files
**Cons:** Paid (starts at $5/month)

**Steps:**
1. Install Heroku CLI
2. Create `Procfile`:
   ```
   web: sh setup.sh && streamlit run multipage/1_Home.py
   ```
3. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```
4. Deploy:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

---

### Option 3: AWS EC2 / Google Cloud / Azure

**Pros:** Full control, scalable
**Cons:** More complex, requires server management

**Basic Steps:**
1. Launch a VM instance
2. SSH into the instance
3. Clone your repository
4. Install Python and dependencies
5. Run Streamlit with:
   ```bash
   streamlit run multipage/1_Home.py --server.port 8501 --server.address 0.0.0.0
   ```
6. Configure firewall to allow port 8501

---

### Option 4: Docker + Any Cloud Platform

**Pros:** Portable, consistent environment
**Cons:** Requires Docker knowledge

**Create `Dockerfile`:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "multipage/1_Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run:**
```bash
docker build -t cataract-detection .
docker run -p 8501:8501 cataract-detection
```

---

## 📝 Pre-Deployment Checklist

- [x] All hardcoded paths converted to relative paths
- [x] requirements.txt created
- [x] .h5 model file in correct location
- [ ] Test all features locally
- [ ] Add .gitignore for venv and __pycache__
- [ ] Update README with deployment instructions
- [ ] Consider using Git LFS for large .h5 file
- [ ] Add error handling for missing files
- [ ] Test with sample images

---

## 🔧 Troubleshooting

### Issue: Model file not found
- Ensure `cataract_vgg19_model.h5` is in `multipage/pages/`
- Check file permissions

### Issue: Out of memory
- Reduce batch size in code
- Use a platform with more RAM

### Issue: Slow loading
- Optimize model loading (cache it)
- Use smaller image sizes

---

## 📊 Current Project Structure
```
Cataract_Detection_with_XAI/
├── multipage/
│   ├── 1_Home.py              # Main entry point
│   └── pages/
│       ├── 2_About.py
│       ├── 3_VGG-19.py
│       ├── 4_XAI.py
│       ├── 5_Dataset.py
│       ├── 6_Working.py       # Detection logic
│       ├── cataract_vgg19_model.h5  # Model file (113 MB)
│       └── model.pkl
├── outputs/                   # Generated images
├── venv/                      # Virtual environment (don't commit)
├── requirements.txt           # Dependencies
├── README.md
└── DEPLOYMENT_GUIDE.md        # This file
```

---

## 🎯 Next Steps

1. **Test locally** - Upload some test images and verify everything works
2. **Choose deployment platform** - Streamlit Cloud is easiest for beginners
3. **Prepare repository** - Clean up and push to GitHub
4. **Deploy** - Follow steps for your chosen platform
5. **Share** - Get your public URL and share your project!

Good luck with your deployment! 🚀

