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


def feature_tf_idf(input_df, gram_range=(1, 5)):
    tf_idf = TfidfVectorizer(sublinear_tf=True,
                            min_df=5, norm='l2',
                            encoding='latin-1',
                            ngram_range=gram_range,
                            stop_words='english')
    feature = tf_idf.fit_transform(input_df['Description'])
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


def recommend_post_neg_feature_importance(input_df):
    # prepare for tf_idf feature with n gram_range
    label = input_df.label
    feature, feature_names = feature_tf_idf(input_df, gram_range=(1, 5))

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
    #len(feature_names), len(random_forest_clf.feature_importances_)
    important_terms = sorted(zip(map(lambda x: round(x, 4), random_forest_clf.feature_importances_), feature_names),
                             reverse=True)

    post_important_terms = [x for x in important_terms
                            if len(get_term_beta(x[1], feature_names_coefficients)) > 0
                            and get_term_beta(x[1], feature_names_coefficients)[0] > 0]

    neg_important_terms = [x for x in important_terms
                            if len(get_term_beta(x[1], feature_names_coefficients)) > 0
                            and get_term_beta(x[1], feature_names_coefficients)[0] < 0]

    return post_important_terms, neg_important_terms


def recommend_post_neg_conf(input_df):
    # prepare for tf_idf feature with n gram_range
    label = input_df.label
    feature, feature_names = feature_tf_idf(input_df)

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

    post = feature_names_coefficients[feature_names_coefficients.coef > 0]
    neg = feature_names_coefficients[feature_names_coefficients.coef < 0]

    post_dict = dict(zip(post.names, post.coef))
    neg_dict = dict(zip(neg.names, neg.coef))

    return post_dict, neg_dict

    return (sorted(post_dict.items(), key=lambda x: x[1], reverse=True),
            sorted(neg_dict.items(), key=lambda x: x[1], reverse=True))


def ranked_frq_words(df):
    feature, feature_names = feature_tf_idf(df)
    feature_tfidf_sum = feature.sum(axis=0).tolist()[0]
    ranks = dict(zip(feature_names, feature_tfidf_sum))

    return ranks

    return sorted(ranks.items(), key=lambda x: x[1], reverse=True)


def recommend(landing_page_raw_text, top=24):
    good_df = similarity.related_sub_df(cleaned_data_df[cleaned_data_df.label == 'Good'],
                                        landing_page_raw_text, 0.75)
    bad_df = similarity.related_sub_df(cleaned_data_df[cleaned_data_df.label == 'Bad'],
                                       landing_page_raw_text, 0.75)
    all_df = good_df.append(bad_df)

    #get all dicts
    ranked_frq_terms_in_good = ranked_frq_words(good_df)
    ranked_frq_terms_in_bad = ranked_frq_words(bad_df)
    post_terms, neg_terms = recommend_post_neg_conf(all_df)

    good = pd.DataFrame({'rank': pd.Series(ranked_frq_terms_in_good), 'conf': pd.Series(post_terms)}).dropna()
    good['score'] = good['rank'] * good['conf']
    bad = pd.DataFrame({'rank': pd.Series(ranked_frq_terms_in_bad), 'conf': pd.Series(neg_terms)}).dropna()
    bad['score'] = bad['rank'] * bad['conf']

    good_dict = good.to_dict('index')
    bad_dict = bad.to_dict('index')

    #good.to_csv('./good.csv')

    goodtop = sorted(good_dict.items(), key=lambda x: x[1]['score'], reverse=True)[:top]
    badtop = sorted(bad_dict.items(), key=lambda x: x[1]['score'])[:top]


    return goodtop, badtop
