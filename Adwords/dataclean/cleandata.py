# import nltk
#
#
# def clean_str(raw):
#     raw_words = nltk.word_tokenize(raw)
#     # Clean words
#     words = [word for word in raw_words if len(word) > 1]
#     words = [word for word in words if word.isalpha()]
#     words = [w.lower() for w in words if w.isalnum()]
#     # Stop words
#     #stopwords = set(nltk.corpus.stopwords.words('english'))
#     #words = [word for word in words if word not in stopwords]
#     # Lemma or Stem; use Lemmatizer for better results
#
#     wnl = nltk.WordNetLemmatizer()
#     cleaned_words = [wnl.lemmatize(t) for t in words]
#     return ' '.join(cleaned_words).strip()