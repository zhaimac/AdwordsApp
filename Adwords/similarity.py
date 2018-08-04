import nltk
import numpy as np

#nltk.download('punkt')
#nltk.download('wordnet')


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
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def get_similarity(dict1, dict2):
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
def add_similarity_feature(df, landing_page_url_raw_text, col):
    landing_pag_url_raw_text_dict = text_to_dict(landing_page_url_raw_text)
    df['Similarity'] = df[col].apply(lambda x: get_similarity(landing_pag_url_raw_text_dict, text_to_dict(x)))
    return df


# Get the Similarity between landing_pag_url_raw_text and each row
# only keep top_similarity
def related_sub_df(df, landing_page_url_raw_text, col, top_similarity=0.8):
    df_with_sim = add_similarity_feature(df, landing_page_url_raw_text, col)
    df_for_page = df_with_sim[df_with_sim.Similarity > df_with_sim.Similarity.quantile(top_similarity)]
    return df_for_page
