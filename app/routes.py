from app import app
from app.forms import AdForm
from flask import json

import Adwords
from flask import render_template, redirect, flash, url_for, request, jsonify


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = AdForm()
    if form.validate_on_submit():  # post and submit validate
        #product = form.product.data
        url = form.landing_page.data

        try:
            landing_page_raw_text = Adwords.fetchLandingPage.url_to_content(url)
        except Exception as e:
            return render_template('index.html', title='AdWords',
                                   form=form,
                                   error="Landing page is void or protected!")
        p, n = Adwords.recommend(landing_page_raw_text)

        if len(landing_page_raw_text) > 1200:
            landing_page_raw_text = landing_page_raw_text[:48] + \
                                    ' ... ' + landing_page_raw_text[200:1200] + '......'
        return render_template('index.html', title='AdWords',
                               form=form,
                               landing_content=landing_page_raw_text,
                               post=p, neg=n)
    # for get or submit not validate
    return render_template('index.html', title='Google Ads Adviser', form=form)  # GET or submit validate Field


@app.route('/api/r', methods=['GET'])
def recommend():
    url = request.args.get('url')
    try:
        landing_page_raw_text = Adwords.fetchLandingPage.url_to_content(url)
    except Exception as e:
        error = dict(
            message="Landing page url is void or protected.",
        )
        response = app.response_class(
            response=json.dumps(error),
            status=400,
            mimetype='application/json'
        )
        return response

    frq_goods, p, n = Adwords.recommend(landing_page_raw_text)
    if len(landing_page_raw_text) > 800:
        landing_page_raw_text = landing_page_raw_text[:48] + \
                                ' ... ' + landing_page_raw_text[200:800] + '...'
    data = dict(
        landing_url=url,
        text=landing_page_raw_text,
        frq_goods=frq_goods,
        post=p,
        neg=n,
    )

    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/bitcoin')
def bitcoin():
    return render_template('bitcoin.html')


@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response
