from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField

class AdForm(FlaskForm):
    landing_page = StringField("Landing Page URL")

    # product = SelectField('Choose a Product', choices=[('window', 'Windows'),
    #                                             ('floor', 'Floor'),
    #                                             ('land', 'Land'),
    #                                             ])
    submit = SubmitField("Generate")