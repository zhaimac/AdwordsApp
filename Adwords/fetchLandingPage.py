import urllib
from bs4 import BeautifulSoup

def landing_page_raw_text(landing_page_url):
    page = urllib.request.urlopen(landing_page_url)
    soup = BeautifulSoup(page.read(), "lxml")
    raw = (soup.get_text())
    return raw