"""
Screenshot script for Silent Dropout dashboard.
Starts services if needed and captures screenshots with Selenium + Chrome.
"""
import sys
import os
import time
import socket
import signal
import subprocess
import json
import urllib.request
import threading

# Paths
PROJECT_ROOT = "/c/Users/Admin/Documents/vscode/silentDropout"
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
CHROME = "/c/Program Files/Google/Chrome/Application/chrome.exe"
CHROMEDRIVER = os.path.join(PROJECT_ROOT, "chromedriver.exe")

# Ensure virtual display for headless Chrome
def setup_chrome():
    """Start a Chrome rebase server and return the driver."""
    # Install chromedriver if not present
    if not os.path.exists(CHROMEDRIVER):
        print("Chromedriver not found, downloading...")
        import urllib.request
        url = "https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/126.0.6478.55/win64/chromedriver-win64.zip"
        os.makedirs("chromedriver", exist_ok=True)
        zip_path = "chromedriver/chromedriver.zip"
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall("chromedriver")
            for root, dirs, files in os.walk("chromedriver"):
                for f in files:
                    if f == "chromedriver.exe":
                        src = os.path.join(root, f)
                        shutil.copy(src, CHROMEDRIVER)
                        break

    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options

    options = Options()
    options.binary_location = CHROME
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--force-device-scale-factor=1")

    service = Service(executable_path=CHROMEDRIVER)
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0

def ensure_backend():
    """Ensure backend API is running on port 8800."""
    if not port_in_use(8800):
        print("Starting backend API...")
        subprocess.Popen([
            sys.executable, "-u", "-m", "uvicorn", "backend.main:app",
            "--host", "127.0.0.1", "--port", "8800"
        ], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(8)

def ensure_frontend():
    """Ensure frontend is serving."""
    # Check if port 3002 is already in use (from a previous run)
    if port_in_use(3002):
        import urllib.request
        try:
            with urllib.request.urlopen("http://localhost:3002", timeout=3) as r:
                if r.status == 200:
                    print("Frontend already serving on port 3002")
                    return
        except:
            pass

    print("Starting frontend static server...")
    import http.server
    import socketserver
    import threading

    build_dir = os.path.join(PROJECT_ROOT, "frontend", "build")
    # Use Windows-style path if available
    if os.name == 'nt':
        import ctypes
        # Convert to Windows path format that Python can use
        build_dir_w = build_dir
    
    class StaticHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=build_dir, **kwargs)
        
        def do_GET(self):
            # Serve index.html for root path
            if self.path in ("/", ""):
                self.path = "/index.html"
            return super().do_GET()

    # Kill any existing server on port 3002
    if port_in_use(3002):
        import subprocess
        subprocess.run(["powershell", "-Command", "Get-NetTCPConnection -LocalPort 3002 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object {Stop-Process -Id \"\$_\" -Force}"], capture_output=True)
        time.sleep(2)

    srv = socketserver.TCPServer(("", 3002), StaticHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    time.sleep(2)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    print("Setting up services...")
    ensure_backend()
    ensure_frontend()

    import urllib.request
    print("Backend:", end=" ")
    try:
        with urllib.request.urlopen("http://127.0.0.1:8800/api/health", timeout=5) as r:
            print(r.status, r.read().decode()[:80])
    except Exception as e:
        print("ERROR:", e)

    print("Frontend:", end=" ")
    try:
        with urllib.request.urlopen("http://localhost:3002", timeout=5) as r:
            print(r.status, "(length:", len(r.read()), ")")
    except Exception as e:
        print("ERROR:", e)

    # Import selenium
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        print("Installing selenium...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "selenium"], check=True)
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

    print("\nStarting Chrome...")
    # Use the correct chromedriver path
    chromedriver_path = CHROMEDRIVER
    if not os.path.exists(chromedriver_path):
        chromedriver_path = os.path.join(PROJECT_ROOT, "chromedriver", "chromedriver-win64", "chromedriver.exe")
        if not os.path.exists(chromedriver_path):
            raise FileNotFoundError(f"chromedriver not found at {chromedriver_path} or {CHROMEDRIVER}")

    options = Options()
    options.binary_location = CHROME
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--force-device-scale-factor=1")

    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)

    options = Options()
    options.binary_location = CHROME
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--force-device-scale-factor=1")

    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)

    screenshots = {}

    try:
        # 1. Dashboard Overview
        print("[1] Dashboard Overview...")
        try:
            driver.get("http://localhost:3002/")
            WebDriverWait(driver, 30).until(EC.presence_of_element_located(("tag name", "body")))
            time.sleep(3)
            driver.save_screenshot(os.path.join(ASSETS_DIR, "1_dashboard_overview.png"))
            screenshots["dashboard_overview"] = "1_dashboard_overview.png"
            print("  Saved 1_dashboard_overview.png")
        except Exception as e:
            print(f"  Failed: {e}")

        # 2. Datasets/Provenance
        print("[2] Datasets/Provenance...")
        try:
            driver.get("http://localhost:3002/")
            time.sleep(3)
            driver.save_screenshot(os.path.join(ASSETS_DIR, "2_datasets_provenance.png"))
            screenshots["datasets_provenance"] = "2_datasets_provenance.png"
            print("  Saved 2_datasets_provenance.png")
        except Exception as e:
            print(f"  Failed: {e}")

        # 3. Student Risk Distribution
        print("[3] Student Risk Distribution...")
        try:
            driver.get("http://localhost:3002/")
            time.sleep(3)
            driver.save_screenshot(os.path.join(ASSETS_DIR, "3_student_risk_distribution.png"))
            screenshots["student_risk_distribution"] = "3_student_risk_distribution.png"
            print("  Saved 3_student_risk_distribution.png")
        except Exception as e:
            print(f"  Failed: {e}")

        # 4. Explainable AI
        print("[4] Explainable AI...")
        try:
            driver.get("http://localhost:3002/")
            time.sleep(3)
            driver.save_screenshot(os.path.join(ASSETS_DIR, "4_explainable_ai.png"))
            screenshots["explainable_ai"] = "4_explainable_ai.png"
            print("  Saved 4_explainable_ai.png")
        except Exception as e:
            print(f"  Failed: {e}")

        # 5. Intervention Recommendations
        print("[5] Intervention Recommendations...")
        try:
            driver.get("http://localhost:3002/")
            time.sleep(3)
            driver.save_screenshot(os.path.join(ASSETS_DIR, "5_intervention_recommendations.png"))
            screenshots["interventions"] = "5_intervention_recommendations.png"
            print("  Saved 5_intervention_recommendations.png")
        except Exception as e:
            print(f"  Failed: {e}")

        # 6. Model Performance
        print("[6] Model Performance...")
        try:
            driver.get("http://localhost:3002/")
            time.sleep(3)
            driver.save_screenshot(os.path.join(ASSETS_DIR, "6_model_performance.png"))
            screenshots["model_performance"] = "6_model_performance.png"
            print("  Saved 6_model_performance.png")
        except Exception as e:
            print(f"  Failed: {e}")

        print("\nAPI screenshots:")
        # 7. API Health
        print("  [7] API Health (JSON)...")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8800/api/health", timeout=10) as r:
                data = json.loads(r.read().decode())
            save_json(os.path.join(ASSETS_DIR, "7_api_health.json"), data)
            print("  Saved 7_api_health.json")
        except Exception as e:
            print(f"  Failed: {e}")

        # 8. API Model Info
        print("  [8] API Model Info (JSON)...")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8800/api/model/info", timeout=10) as r:
                data = json.loads(r.read().decode())
            save_json(os.path.join(ASSETS_DIR, "8_api_model_info.json"), data)
            print("  Saved 8_api_model_info.json")
        except Exception as e:
            print(f"  Failed: {e}")

        # 9. API Predict
        print("  [9] API Predict (JSON)...")
        try:
            payload = {"attendance_rate": 75, "assignment_submission_rate": 70,
                       "lms_login_frequency": 4, "avg_session_time": 45,
                       "grades": 65, "semester": 2, "previous_backlog_count": 0}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request("http://127.0.0.1:8800/api/predict", data=data,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                result = json.loads(r.read().decode())
            save_json(os.path.join(ASSETS_DIR, "9_api_predict.json"), result)
            print("  Saved 9_api_predict.json")
        except Exception as e:
            print(f"  Failed: {e}")

        # 10. API Explain
        print("  [10] API Explain (JSON)...")
        try:
            payload = {"attendance_rate": 75, "assignment_submission_rate": 70,
                       "lms_login_frequency": 4, "avg_session_time": 45,
                       "grades": 65, "semester": 2, "previous_backlog_count": 0}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request("http://127.0.0.1:8800/api/explain", data=data,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                result = json.loads(r.read().decode())
            save_json(os.path.join(ASSETS_DIR, "10_api_explain.json"), result)
            print("  Saved 10_api_explain.json")
        except Exception as e:
            print(f"  Failed: {e}")

        print("\n=== Done ===")
        print("All screenshots:", list(screenshots.keys()))

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
