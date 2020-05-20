from flask_table import Table, Col, ButtonCol


class DataTable(Table):
        classes = ['table', 'table-striped', 'table-condensed', 'table-hover']
        _id = Col('Id', show=False)
        ModelName = Col('Назва моделі')
        TopWords = Col('Кількість найчастіших слів')
        MaxReviewLen = Col('Довжина тексту')
        Configuration = Col('Конфігурація моделі')
        ModelFileName = Col('Назва файлу з моделлю')
        edit = ButtonCol('Вибрати модель', 'choose', url_kwargs=dict(id='_id'))

class ModelTable(Table):
        classes = ['table', 'table-striped', 'table-condensed', 'table-hover']
        _id = Col('Id', show=False)
        ModelName = Col('Назва моделі')
        TopWords = Col('Кількість найчастіших слів')
        MaxReviewLen = Col('Довжина тексту')
        Configuration = Col('Конфігурація моделі')
        ModelFileName = Col('Назва файлу з моделлю')
