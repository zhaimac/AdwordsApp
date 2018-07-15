import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier

from Adwords import cleandata
from Adwords import similarity
from Adwords import nlp_feature


text_attribs = ["add_define_text"]
cat_attribs = ['Campaign']

tfidf_pipline = Pipeline([
    ('selector', nlp_feature.DataFrameSelector(text_attribs)),
    #('vect', CountVectorizer()),
    #('tfidf', TfidfTransformer()),
    ('tfidf', TfidfVectorizer(stop_words = 'english', ngram_range=(1,6))),
])

cat_pipline = Pipeline([
    ('selector', nlp_feature.DataFrameSelector(cat_attribs)),
    ('l', nlp_feature.MyLabelBinarizer()),
    ('onehot', nlp_feature.OneHotEncoder()),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("tfidf_pipline", tfidf_pipline),
    #("cat_pipline", cat_pipline),
])


# GET Train Data
df0 = pd.read_csv('./app/Product_Labeled_Ad_Data.csv')
df = cleandata.clean_data_1(df0)


def gen_add_def(landing_pag_url):
    landing_page_url_raw_text = similarity.url_to_raw_content(landing_pag_url)
    df_for_page = similarity.related_sub_df(df, landing_page_url_raw_text)
    ad_text_df = nlp_feature.add_label(df_for_page)

    # split data to train and test
    msk = np.random.rand(len(ad_text_df)) < 0.8
    ad_text_df_train = ad_text_df[msk]
    ad_text_df_test = ad_text_df[~msk]

    full_pipeline.fit(ad_text_df)

    data_train = full_pipeline.fit_transform(ad_text_df_train)
    data_test = full_pipeline.transform(ad_text_df_test)


    # Supervised learning model
    text_clf = Pipeline([
        ('rfr', RandomForestClassifier())
    ])
    text_clf.fit( data_train, ad_text_df_train.Label)

    y_pred = text_clf.predict(data_test)

    print(accuracy_score(ad_text_df_test.Label, y_pred))
    print(confusion_matrix(ad_text_df_test.Label, y_pred))

    forest = text_clf.named_steps['rfr']

    names = tfidf_pipline.named_steps['tfidf'].get_feature_names()
    len(names), len(forest.feature_importances_)
    return sorted(zip(map(lambda x: round(x, 4), forest.feature_importances_), names), reverse=True)[:100]