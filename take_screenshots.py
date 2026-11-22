"""
Taking screenshots of the Silent Dropout dashboard using Chrome headless.
Requires Selenium: pip install selenium
"""
import time
import json
import urllib.request
import subprocess
import sys
import os
import signal
import socket

# Paths
CHROME = "/c/Program Files/Google/Chrome/Application/chrome.exe"
CHROMEDRIVER = "C:/Users/Admin/Documents/vscode/silentDropout/chromedriver.exe"
PROJECT_ROOT = "C:/Users/Admin/Documents/vscode/silentDropout"
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Screenshot URLs - use the running services
BACKEND_URL = "http://127.0.0.1:8800"
FRONTEND_URL = "http://localhost:3001"  # Static build server

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0

def ensure_services():
    """Ensure backend and frontend are running, restart if needed."""
    if not check_port(8800):
        print("Starting backend API...")
        subprocess.Popen(
            [sys.executable, "-u", "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8800"],
            cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(8)

    if check_port(3001):
        print("Frontend already serving on port 3001")
    else:
        print("Starting frontend static server...")
        import http.server
        import socketserver
        class StaticHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=os.path.join(PROJECT_ROOT, "frontend", "build"), **kwargs)
        with socketserver.TCPServer(("", 3001), StaticHandler) as httpd:
            httpd.serve_forever()

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def take_screenshot(driver, name):
    filepath = os.path.join(ASSETS_DIR, name)
    driver.save_screenshot(filepath)
    print(f"  Saved: {filepath}")
    return filepath

def main():
    print("Checking services...")
    ensure_services()

    # Install selenium if needed
    try:
        import selenium
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException
    except ImportError:
        print("Installing selenium...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "selenium"], check=True)
        import selenium
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException

    # Configure Chrome options
    options = Options()
    options.binary_location = CHROME
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--force-device-scale-factor=1")

    service = Service(executable_path=CHROMEDRIVER)
    driver = webdriver.Chrome(service=service, options=options)

    screenshots = {}

    print("\n=== Taking screenshots ===\n")

    # 1. Dashboard Overview
    print("[1/6] Dashboard Overview...")
    try:
        driver.get(FRONTEND_URL + "/")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located(("css selector", "body"))
        )
        time.sleep(3)
        take_screenshot(driver, "1_dashboard_overview.png")
        screenshots["dashboard_overview"] = "1_dashboard_overview.png"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["dashboard_overview"] = None

    # 2. Datasets/Provenance
    print("[2/6] Datasets/Provenance...")
    try:
        driver.get(FRONTEND_URL + "/")
        # Click on datasets section if nav exists
        time.sleep(2)
        # Try to find nav items
        for selector in ["[data-testid='datasets']", "nav a", ".nav-link", "li:not(.active) a"]:
            try:
                el = driver.find_element("css selector", selector)
                if el.text.strip() and "datasets" in el.text.lower():
                    el.click()
                    break
            except:
                pass
        time.sleep(3)
        take_screenshot(driver, "2_datasets_provenance.png")
        screenshots["datasets_provenance"] = "2_datasets_provenance.png"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["datasets_provenance"] = None

    # 3. Student Risk Distribution
    print("[3/6] Student Risk Distribution...")
    try:
        driver.get(FRONTEND_URL + "/")
        time.sleep(3)
        # Try to navigate to risk distribution
        for selector in ["[data-testid='risk-distribution']", ".nav-link[href*='risk']", "li:not(.active) a"]:
            try:
                el = driver.find_element("css selector", selector)
                if el.text.strip() and ("risk" in el.text.lower() or "risk" in el.get_attribute("href").lower()):
                    el.click()
                    break
            except:
                pass
        time.sleep(3)
        take_screenshot(driver, "3_risk_distribution.png")
        screenshots["risk_distribution"] = "3_risk_distribution.png"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["risk_distribution"] = None

    # 4. Explainable AI
    print("[4/6] Explainable AI...")
    try:
        driver.get(FRONTEND_URL + "/")
        time.sleep(3)
        for selector in ["[data-testid='explainable-ai']", ".nav-link[href*='explain']", "li:not(.active) a"]:
            try:
                el = driver.find_element("css selector", selector)
                if el.text.strip() and ("explain" in el.text.lower() or "xai" in el.text.lower()):
                    el.click()
                    break
            except:
                pass
        time.sleep(3)
        take_screenshot(driver, "4_explainable_ai.png")
        screenshots["explainable_ai"] = "4_explainable_ai.png"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["explainable_ai"] = None

    # 5. Intervention Recommendations
    print("[5/6] Intervention Recommendations...")
    try:
        driver.get(FRONTEND_URL + "/")
        time.sleep(3)
        for selector in ["[data-testid='interventions']", ".nav-link[href*='intervention']", "li:not(.active) a"]:
            try:
                el = driver.find_element("css selector", selector)
                if el.text.strip() and ("intervention" in el.text.lower() or "intervene" in el.text.lower()):
                    el.click()
                    break
            except:
                pass
        time.sleep(3)
        take_screenshot(driver, "5_interventions.png")
        screenshots["interventions"] = "5_interventions.png"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["interventions"] = None

    # 6. Model Performance
    print("[6/6] Model Performance...")
    try:
        driver.get(FRONTEND_URL + "/")
        time.sleep(3)
        for selector in ["[data-testid='model-performance']", ".nav-link[href*='model']", "li:not(.active) a"]:
            try:
                el = driver.find_element("css selector", selector)
                if el.text.strip() and ("perform" in el.text.lower() or "model" in el.text.lower()):
                    el.click()
                    break
            except:
                pass
        time.sleep(3)
        take_screenshot(driver, "6_model_performance.png")
        screenshots["model_performance"] = "6_model_performance.png"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["model_performance"] = None

    # API screenshots
    print("\n=== API screenshots ===\n")

    # 7. API Health
    print("[7/7] API Health (backend)...")
    try:
        with urllib.request.urlopen(BACKEND_URL + "/api/health", timeout=10) as r:
            data = json.loads(r.read().decode())
        filepath = os.path.join(ASSETS_DIR, "7_api_health.json")
        save_json(filepath, data)
        print(f"  Saved: {filepath}")
        screenshots["api_health"] = "7_api_health.json"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["api_health"] = None

    # 8. API Model Info
    print("[8/8] API Model Info (backend)...")
    try:
        with urllib.request.urlopen(BACKEND_URL + "/api/model/info", timeout=10) as r:
            data = json.loads(r.read().decode())
        filepath = os.path.join(ASSETS_DIR, "8_api_model_info.json")
        save_json(filepath, data)
        print(f"  Saved: {filepath}")
        screenshots["api_model_info"] = "8_api_model_info.json"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["api_model_info"] = None

    # 9. API Predict
    print("Requesting /api/predict...")
    try:
        payload = {"attendance_rate": 75, "assignment_submission_rate": 70,
                   "lms_login_frequency": 4, "avg_session_time": 45,
                   "grades": 65, "semester": 2, "previous_backlog_count": 0}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(BACKEND_URL + "/api/predict", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
        filepath = os.path.join(ASSETS_DIR, "9_api_predict.json")
        save_json(filepath, result)
        print(f"  Saved: {filepath}")
        screenshots["api_predict"] = "9_api_predict.json"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["api_predict"] = None

    # 10. API Explain
    print("Requesting /api/explain...")
    try:
        payload = {"attendance_rate": 75, "assignment_submission_rate": 70,
                   "lms_login_frequency": 4, "avg_session_time": 45,
                   "grades": 65, "semester": 2, "previous_backlog_count": 0}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(BACKEND_URL + "/api/explain", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
        filepath = os.path.join(ASSETS_DIR, "10_api_explain.json")
        save_json(filepath, result)
        print(f"  Saved: {filepath}")
        screenshots["api_explain"] = "10_api_explain.json"
    except Exception as e:
        print(f"  Failed: {e}")
        screenshots["api_explain"] = None

    driver.quit()
    print("\n=== Done ===")
    print("Screenshots:", list(screenshots.keys()))

if __name__ == "__main__":
    main()
