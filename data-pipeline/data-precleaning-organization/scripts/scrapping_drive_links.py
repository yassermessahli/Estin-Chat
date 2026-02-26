from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time

# Setup Chrome options for headless mode
chrome_options = Options()
chrome_options.add_argument("--headless=new")  # use new headless mode (more stable)
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--remote-debugging-port=9222")
chrome_options.add_argument("--disable-software-rasterizer")
chrome_options.add_argument("--user-data-dir=/tmp/chrome-profile")

# Use system's chromedriver
driver = webdriver.Chrome(service=Service("/usr/bin/chromedriver"), options=chrome_options)

# Test if working
driver.get("https://estin-student.vercel.app")
time.sleep(5)

print(driver.title)  # Should print: ESTIN Student

# Save HTML to inspect
with open("output.html", "w", encoding="utf-8") as f:
    f.write(driver.page_source)

driver.quit()
