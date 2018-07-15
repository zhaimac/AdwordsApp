import nltk
import numpy as np
import urllib
from bs4 import BeautifulSoup
import re

#nltk.download('punkt')
#nltk.download('wordnet')

def url_to_raw_content(page_url):
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


def text_to_dict(raw):
    raw_words = nltk.word_tokenize(raw)
    # Clean words
    words = [word for word in raw_words if len(word) > 1]
    words = [word for word in words if word.isalpha()]
    words = [w.lower() for w in words if w.isalnum()]

    # Stop words
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stopwords]

    # Lemma or Stem; use Lemmatizer for better results
    wnl = nltk.WordNetLemmatizer()
    cleaned_words = [wnl.lemmatize(t) for t in words]

    count = nltk.defaultdict(int)
    for w in cleaned_words:
        count[w] += 1
    return count


def cos_sim(a, b):
    dot_priduct = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_priduct / (norm_a * norm_b)


def getSimilarity(dict1, dict2):
    all_words_list = []
    for key in dict1:
        all_words_list.append(key)
    for key in dict2:
        all_words_list.append(key)
    all_words_list_size = len(all_words_list)

    v1 = np.zeros(all_words_list_size, dtype=np.int)
    v2 = np.zeros(all_words_list_size, dtype=np.int)

    i = 0
    for (key) in all_words_list:
        v1[i] = dict1.get(key, 0)
        v2[i] = dict2.get(key, 0)
        i = i + 1

    return cos_sim(v1, v2)


# Get the Similarity between landing_pag_url_raw_text and each row
# The next step will take some time
def related_sub_df(df, landing_page_url_raw_text):
    landing_pag_url_raw_text_dict = text_to_dict(landing_page_url_raw_text)
    df['Similarity'] = df.add_define_text.apply(lambda x:getSimilarity(landing_pag_url_raw_text_dict, text_to_dict(x)))

    df_for_page = df[df.Similarity > df.Similarity.quantile(.8)]
    return df_for_page