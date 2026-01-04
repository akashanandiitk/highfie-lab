# HighFIE Lab - Deployment Guide

## 🎯 Goal
Make HighFIE Lab publicly accessible at: `https://home.iitk.ac.in/~akasha/highfie/`

---

## 📋 Prerequisites

Before starting, ensure you have:
- [ ] SSH access to IIT Kanpur server
- [ ] Your app files: `streamlit_app_phase4.py` and `fd_coefficients.json`
- [ ] Python 3.8+ installed on the server
- [ ] Permission to run long-running processes

---

## 🚀 Recommended Deployment: Streamlit Community Cloud (EASIEST)

### Option 1A: Deploy to Streamlit Cloud + Embed in Your Webpage

**Advantages:**
- ✅ Free hosting
- ✅ Auto-updates when you push to GitHub
- ✅ HTTPS included
- ✅ No server maintenance
- ✅ Can embed in your IIT Kanpur webpage

**Steps:**

### 1. Create GitHub Repository

```bash
# On your local machine
cd /path/to/your/project
git init
git add streamlit_app_phase4.py fd_coefficients.json
git commit -m "Initial commit - HighFIE Lab"

# Create repository on GitHub (github.com/new)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/highfie-lab.git
git push -u origin main
```

### 2. Create `requirements.txt`

Create a file named `requirements.txt` in your repository:

```txt
streamlit>=1.28.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
sympy>=1.12
```

### 3. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select:
   - Repository: `YOUR_USERNAME/highfie-lab`
   - Branch: `main`
   - Main file: `streamlit_app_phase4.py`
5. Click "Deploy"

**Your app will be live at:** `https://YOUR_USERNAME-highfie-lab-streamlit-app-phase4-xxxxx.streamlit.app`

### 4. Embed in Your IIT Kanpur Webpage

Add this to your webpage (`https://home.iitk.ac.in/~akasha/highfie.html`):

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HighFIE Lab - High-Order Fourier Interpolation with Extension</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #fff8e7;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2em;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }
        .container {
            width: 100%;
            height: calc(100vh - 100px);
            border: none;
        }
        .footer {
            background-color: #34495e;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 HighFIE Lab</h1>
        <p>High-Order Fourier Interpolation with Extension</p>
    </div>
    
    <iframe 
        src="https://YOUR_USERNAME-highfie-lab-streamlit-app-phase4-xxxxx.streamlit.app?embedded=true"
        class="container"
        frameborder="0"
        scrolling="yes"
        allowfullscreen>
    </iframe>
    
    <div class="footer">
        Developed at <strong>Indian Institute of Technology Kanpur</strong> | 
        <a href="https://home.iitk.ac.in/~akasha/" style="color: #3498db;">Dr. Akash Anand</a>
    </div>
</body>
</html>
```

**Upload this HTML file to your server:**
```bash
scp highfie.html akasha@server.iitk.ac.in:~/public_html/
```

---

## 🖥️ Option 1B: Host Directly on IIT Kanpur Server

**Advantages:**
- ✅ Full control
- ✅ Hosted on IIT domain
- ✅ No third-party dependency

**Disadvantages:**
- ❌ Requires server setup
- ❌ Need to manage process
- ❌ May need admin help for port access

### Steps:

### 1. Upload Files to Server

```bash
# Create directory on server
ssh akasha@server.iitk.ac.in
mkdir -p ~/apps/highfie-lab
exit

# Upload files
scp streamlit_app_phase4.py akasha@server.iitk.ac.in:~/apps/highfie-lab/
scp fd_coefficients.json akasha@server.iitk.ac.in:~/apps/highfie-lab/
```

### 2. Set Up Python Environment on Server

```bash
# SSH into server
ssh akasha@server.iitk.ac.in

# Navigate to app directory
cd ~/apps/highfie-lab

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install streamlit numpy matplotlib scipy sympy
```

### 3. Test the App Locally

```bash
# Still on server
streamlit run streamlit_app_phase4.py --server.port 8501
```

Visit in browser: `http://server.iitk.ac.in:8501` (if port is accessible)

### 4. Set Up as Background Service

Create `~/apps/highfie-lab/start_highfie.sh`:

```bash
#!/bin/bash
cd ~/apps/highfie-lab
source venv/bin/activate
nohup streamlit run streamlit_app_phase4.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > highfie.log 2>&1 &
echo $! > highfie.pid
echo "HighFIE Lab started on port 8501"
```

Create `~/apps/highfie-lab/stop_highfie.sh`:

```bash
#!/bin/bash
if [ -f ~/apps/highfie-lab/highfie.pid ]; then
    kill $(cat ~/apps/highfie-lab/highfie.pid)
    rm ~/apps/highfie-lab/highfie.pid
    echo "HighFIE Lab stopped"
else
    echo "No PID file found"
fi
```

Make executable:
```bash
chmod +x start_highfie.sh stop_highfie.sh
```

### 5. Configure Reverse Proxy (Need Admin Help)

Ask your system admin to set up Apache/Nginx reverse proxy:

**Apache configuration** (`/etc/apache2/sites-available/highfie.conf`):

```apache
<Location /~akasha/highfie>
    ProxyPass http://localhost:8501
    ProxyPassReverse http://localhost:8501
    ProxyPreserveHost On
</Location>
```

**Nginx configuration**:

```nginx
location /~akasha/highfie/ {
    proxy_pass http://localhost:8501/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

### 6. Start the Application

```bash
cd ~/apps/highfie-lab
./start_highfie.sh
```

Your app will be accessible at: `https://home.iitk.ac.in/~akasha/highfie/`

---

## 🐳 Option 2: Docker Deployment (ADVANCED)

If your server supports Docker:

### 1. Create `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    streamlit>=1.28.0 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    scipy>=1.10.0 \
    sympy>=1.12

# Copy application files
COPY streamlit_app_phase4.py .
COPY fd_coefficients.json .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
ENTRYPOINT ["streamlit", "run", "streamlit_app_phase4.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build and Run

```bash
# Build image
docker build -t highfie-lab .

# Run container
docker run -d -p 8501:8501 --name highfie-lab highfie-lab
```

---

## 📝 Adding to Your Main Webpage

Update your main page (`https://home.iitk.ac.in/~akasha/index.html`) to link to HighFIE Lab:

```html
<div class="project">
    <h3>🔬 HighFIE Lab</h3>
    <p>
        An interactive web interface for developing and testing high-order 
        Fourier Interpolation with grid extension methods.
    </p>
    <p>
        <a href="highfie.html" class="btn">Launch HighFIE Lab</a>
        <a href="https://github.com/YOUR_USERNAME/highfie-lab" class="btn-secondary">View on GitHub</a>
    </p>
</div>
```

---

## 🔒 Security Considerations

### If hosting on IIT server:

1. **Firewall Rules**: Ensure port 8501 is only accessible through reverse proxy
2. **HTTPS**: Ensure Apache/Nginx handles SSL
3. **Resource Limits**: Set memory/CPU limits to prevent abuse
4. **Authentication** (optional): Add basic auth if needed

```python
# In streamlit_app_phase4.py, at the top:
import streamlit as st

def check_password():
    """Returns True if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == "YOUR_PASSWORD":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()

# Rest of your app code...
```

---

## 📊 Monitoring

### Check if app is running:

```bash
# Check process
ps aux | grep streamlit

# Check logs
tail -f ~/apps/highfie-lab/highfie.log

# Check if port is listening
netstat -tlnp | grep 8501
```

### Auto-restart on server reboot:

Add to crontab (`crontab -e`):

```bash
@reboot /home/akasha/apps/highfie-lab/start_highfie.sh
```

---

## 🎨 Customization for Your Webpage

### Match Your Website Theme

If you want the app to match your website's navigation/header exactly:

**Option A**: Modify `streamlit_app_phase4.py` to include custom header:

```python
# Add after imports
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Add your custom header */
    .custom-header {
        background-color: #2c3e50;
        color: white;
        padding: 20px;
        margin: -70px -70px 20px -70px;
    }
</style>
<div class="custom-header">
    <h1>HighFIE Lab</h1>
    <p>Part of Dr. Akash Anand's Research Tools</p>
</div>
""", unsafe_allow_html=True)
```

**Option B**: Use iframe with header (as shown in Option 1A)

---

## 🐛 Troubleshooting

### App won't start:
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check dependencies
pip list | grep streamlit

# Run in foreground to see errors
streamlit run streamlit_app_phase4.py
```

### Port already in use:
```bash
# Find what's using the port
lsof -i :8501

# Kill the process
kill -9 PID
```

### Can't access from outside:
- Check firewall rules
- Verify reverse proxy configuration
- Check server logs: `/var/log/apache2/error.log` or `/var/log/nginx/error.log`

---

## ✅ Recommended Approach for You

**For quickest deployment with minimal hassle:**

1. **Use Streamlit Community Cloud** (Option 1A)
   - Create GitHub repo
   - Deploy to Streamlit Cloud
   - Embed in your IIT webpage
   - **Time: ~30 minutes**

**For full control on IIT infrastructure:**

2. **Contact IIT system admin** and request:
   - Permission to run Streamlit on port 8501
   - Reverse proxy setup for `/~akasha/highfie`
   - Then follow Option 1B
   - **Time: ~2 hours + admin approval time**

---

## 📧 Next Steps

1. **Choose deployment method** (I recommend Option 1A for simplicity)
2. **Create GitHub repository** if using Streamlit Cloud
3. **Test deployment** with the provided steps
4. **Update your main webpage** to link to HighFIE Lab
5. **Share the link!** 🎉

---

## 🎓 Example Live URL Structures

**Streamlit Cloud + Embedded:**
- App URL: `https://akashanand-highfie-lab.streamlit.app`
- Your page: `https://home.iitk.ac.in/~akasha/highfie.html`
- Embedded: iframe shows Streamlit app

**IIT Server Direct:**
- Direct URL: `https://home.iitk.ac.in/~akasha/highfie/`
- Cleaner, but needs admin setup

---

## 📞 Need Help?

If you run into issues:
1. Check Streamlit documentation: https://docs.streamlit.io/
2. GitHub issues for Streamlit: https://github.com/streamlit/streamlit/issues
3. IIT Kanpur IT support for server access issues

---

**Ready to deploy? Choose your option and let me know if you need help with any step!** 🚀
