from app import app
from flask import render_template, redirect, flash, url_for

import Adwords

from app.forms import AdForm
from bs4 import BeautifulSoup
import urllib.request

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = AdForm()
    if form.validate_on_submit():  # post and submit validate
        product = form.product.data
        landing_page_url = form.landing_page.data
        print(product, landing_page_url)

        page = urllib.request.urlopen(landing_page_url)
        soup = BeautifulSoup(page.read(), "lxml")
        raw = (soup.get_text())

        hot_gram = Adwords.gen_add_def(landing_page_url)

        return render_template('index.html', title='AdWords',
                               form=form,
                               result_title="Recommend Phrase for AdWord",
                               landing_content=raw[:800] + '...' ,
                               hot_gram=hot_gram)

    # for get or submit not validate
    return render_template('index.html', title='AdWords', form=form)  # GET or submit validate Flaid


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/bitcoin')
def bitcoin():
    return render_template('bitcoin.html')