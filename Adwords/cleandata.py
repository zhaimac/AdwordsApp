import nltk

def clean_str(raw):
    raw_words = nltk.word_tokenize(raw)
    # Clean words
    words = [word for word in raw_words if len(word) > 1]
    words = [word for word in words if word.isalpha()]
    words = [w.lower() for w in words if w.isalnum()]
    # Stop words
    #stopwords = set(nltk.corpus.stopwords.words('english'))
    #words = [word for word in words if word not in stopwords]
    # Lemma or Stem; use Lemmatizer for better results
    wnl = nltk.WordNetLemmatizer()
    cleaned_words = [wnl.lemmatize(t) for t in words]
    return ' '.join(cleaned_words).strip()

def clean_data_1(df0):
    df = df0.copy()
    df = df[df.Product == 'Window']

    df['Headline']= df['Headline'].fillna('') + ' ' +df['Headline 1'].fillna('') + ' ' + df['Headline 2'].fillna('')
    df['Headline']= df['Headline'].map(lambda x: clean_str(x))
    del df['Headline 1']
    del df['Headline 2']

    df['Description']= df['Description'].fillna('') + ' ' +df['Description line 1'].fillna('') + ' ' + df['Description line 2'].fillna('')
    df['Description']= df['Description'].map(lambda x: clean_str(x))
    del df['Description line 1']
    del df['Description line 2']

    df[(df['Short headline'].isnull()) & (df['Short headline'] != ' --')]
    del df['Short headline']
    df[(df['Long headline'].isnull()) & (df['Long headline'] != ' --')]
    del df['Long headline']

    df['add_define_text'] = (df.Headline + ' ' + df.Description)
    return df