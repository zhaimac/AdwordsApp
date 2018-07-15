from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#Create a class to select numerical or categorical columns
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values.ravel()


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x).reshape(-1, 1)

# Adding label to indentify if a add define is good
def add_label(df_for_page):
    ad_text_df = df_for_page[['add_define_text', 'CTR', 'Campaign', 'Product']]
    ad_text_df = ad_text_df.dropna()
    ad_text_df.drop_duplicates(subset='add_define_text', keep="last", inplace=True)
    ad_text_df = ad_text_df[ad_text_df['CTR']>0]

    print(ad_text_df.shape)

    def ctr2label(ctr):
        if ctr<0.019555:
            return "Bad"
        if ctr<0.069555: #<0.069555
            return "Mid"
        elif ctr<0.108429: #<0.108429:
            return "OK"
        return "GOOD"
    ad_text_df['Label'] = ad_text_df['CTR'].apply(ctr2label)

    # only consider Bad and GOOD, Remove margin
    ad_text_df = ad_text_df[ (ad_text_df['Label']=="GOOD") | (ad_text_df['Label']=="Bad")]
    return ad_text_df