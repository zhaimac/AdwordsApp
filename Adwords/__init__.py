import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from Adwords.dataclean import cleandata
from Adwords import similarity
from Adwords import fetchLandingPage

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# pre_reload cleaned_data
cleaned_data_df = pd.read_csv('./app/data_cleaned_v8.csv')


def col_to_tf_idf_ngram(input_df, col_name, gram_range=(1, 5)):
    tf_idf = TfidfVectorizer(sublinear_tf=True,
                            min_df=5, norm='l2',
                            encoding='latin-1',
                            ngram_range=gram_range,
                            stop_words='english')
    feature = tf_idf.fit_transform(input_df[col_name])
    feature_names = tf_idf.get_feature_names()
    return feature, feature_names


def get_feature_names_coefficients(coefficients, feature_names):
    feature_df = pd.DataFrame({"names": feature_names})
    coefficients_df = pd.DataFrame(coefficients)
    coefficients_df = pd.DataFrame(coefficients_df.T)
    coefficients_df.columns = ['coef']

    feature_names_coefficients = feature_df
    feature_names_coefficients['names'] = feature_df.names
    feature_names_coefficients['coef'] = coefficients_df.coef
    return feature_names_coefficients


def get_term_beta(term, feature_names_coefficients):
    beta = feature_names_coefficients[feature_names_coefficients.names == term].coef.values
    return beta


# this one not using anymore but keep here
def recommend_post_neg_feature_importance(input_df):
    # prepare for tf_idf feature with n gram_range
    label = input_df.label
    col = 'Description'
    feature, feature_names = col_to_tf_idf_ngram(input_df, col)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(feature, label, test_size=0.2, random_state=42)

    # svc to get coefficients
    svc = LinearSVC()
    svc.fit(features_train, labels_train)
    y_prd = svc.predict(features_test)

    conf_mat = confusion_matrix(labels_test, y_prd)
    print(accuracy_score(labels_test, y_prd))
    print(conf_mat)

    # Link back svc coefficients to feature name
    feature_names_coefficients = get_feature_names_coefficients(svc.coef_, feature_names)

    # random_forest_clf to get important_terms
    random_forest_clf = RandomForestClassifier()
    random_forest_clf.fit(features_train, labels_train)
    y_prd = random_forest_clf.predict(features_test)

    conf_mat = confusion_matrix(labels_test, y_prd)
    print(accuracy_score(labels_test, y_prd))
    print(conf_mat)

    # calculate post_important_terms and neg_important_terms
    important_terms = sorted(zip(map(lambda x: round(x, 4), random_forest_clf.feature_importances_), feature_names),
                             reverse=True)

    post_important_terms = [x for x in important_terms
                            if len(get_term_beta(x[1], feature_names_coefficients)) > 0
                            and get_term_beta(x[1], feature_names_coefficients)[0] > 0]

    neg_important_terms = [x for x in important_terms
                            if len(get_term_beta(x[1], feature_names_coefficients)) > 0
                            and get_term_beta(x[1], feature_names_coefficients)[0] < 0]

    return post_important_terms, neg_important_terms


def recommend_post_neg_conf(input_df, col):
    label = input_df.label
    feature, feature_names = col_to_tf_idf_ngram(input_df, col)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(feature, label, test_size=0.2, random_state=42)

    # svc to get coefficients
    svc = LinearSVC()
    svc.fit(features_train, labels_train)
    y_prd = svc.predict(features_test)

    conf_mat = confusion_matrix(labels_test, y_prd)
    accuracy = accuracy_score(labels_test, y_prd)

    # Link back svc coefficients to feature name
    feature_names_coefficients = get_feature_names_coefficients(svc.coef_, feature_names)

    post = feature_names_coefficients[feature_names_coefficients.coef > 0]
    neg = feature_names_coefficients[feature_names_coefficients.coef < 0]
    post_dict = dict(zip(post.names, post.coef))
    neg_dict = dict(zip(neg.names, neg.coef))

    # random_forest_clf to get feature_importance
    random_forest_clf = RandomForestClassifier(max_features=4)
    random_forest_clf.fit(features_train, labels_train)
    feature_importance = dict(zip(feature_names, random_forest_clf.feature_importances_))

    return post_dict, neg_dict, feature_importance, conf_mat, accuracy


def ranked_frq_words(df, col):
    feature, feature_names = col_to_tf_idf_ngram(df, col)
    feature_tf_idf_sum = feature.sum(axis=0).tolist()[0]
    return dict(zip(feature_names, feature_tf_idf_sum))


def sigmoid(x):
    return x/(1 + abs(x))


def recommend_by_col(landing_page_raw_text, col, top=24):
    good_df = similarity.related_sub_df(cleaned_data_df[cleaned_data_df.label == 'Good'],
                                        landing_page_raw_text, col, 0.75)
    bad_df = similarity.related_sub_df(cleaned_data_df[cleaned_data_df.label == 'Bad'],
                                       landing_page_raw_text, col, 0.75)
    all_df = good_df.append(bad_df)

    # get all dicts
    ranked_frq_terms_in_good = ranked_frq_words(good_df, col)
    ranked_frq_terms_in_bad = ranked_frq_words(bad_df, col)
    post_terms, neg_terms, importance, conf_mat, accuracy = recommend_post_neg_conf(all_df, col)

    good = pd.DataFrame({'rank': pd.Series(ranked_frq_terms_in_good),
                         'conf': pd.Series(post_terms),
                         'imp': pd.Series(importance)}).dropna()
    bad = pd.DataFrame({'rank': pd.Series(ranked_frq_terms_in_bad),
                        'conf': pd.Series(neg_terms),
                        'imp': pd.Series(importance)}).dropna()

    good['score'] = sigmoid(good['rank'] * good['rank'] * good['conf'] * good['imp'] * 100)
    bad['score'] = sigmoid(bad['rank'] * bad['rank'] * bad['conf'] * bad['imp'] * 100)

    good_dict = good.to_dict('index')
    bad_dict = bad.to_dict('index')
    #good.to_csv('./results/good_' + col + '.csv')
    #bad.to_csv('./results/bad_' + col + '.csv')

    good_tops = sorted(good_dict.items(), key=lambda x: x[1]['score'], reverse=True)[:top]
    bad_tops = sorted(bad_dict.items(), key=lambda x: x[1]['score'])[:top]

    return good_tops, bad_tops, conf_mat.tolist(), accuracy
