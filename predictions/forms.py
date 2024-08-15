from django.forms import *


class FileForms(Form):
    file = FileField(
        help_text='Select .xlsx File',
        widget=FileInput(attrs={'accept': '.xlsx'})
    )
