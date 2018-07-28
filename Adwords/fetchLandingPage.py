import urllib
import re
from bs4 import BeautifulSoup


def url_to_content(page_url):
    page = urllib.request.urlopen(page_url)
    soup = BeautifulSoup(page.read(), "lxml")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()  # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return re.sub('\s+', ' ', text).strip()