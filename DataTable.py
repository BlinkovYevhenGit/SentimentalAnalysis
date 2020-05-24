from flask_table import Table, Col, ButtonCol, NestedTableCol


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

class ResultTable(Table):
        classes = ['table', 'table-striped', 'table-condensed', 'table-hover']
        Text = Col('Назва моделі')
        Prediction = Col('Значення тональності')
        PredictionClass = Col('Визначений клас тексту')

class StatisticsTable(Table):
        classes = ['table', 'table-striped', 'table-condensed', 'table-hover']
        EpochNumber=Col('Номер епохи')
        Accuracy=Col('Точність моделі')
        Loss=Col('Значення функції втрат')
class SubStageTable(Table):
        classes = ['table', 'table-striped', 'table-condensed', 'table-hover']
        StageName=Col("Назва етапу")
        subStatisticsTable=NestedTableCol('Статистика моделі',StatisticsTable)
class ReportTable(Table):
        classes = ['table', 'table-striped', 'table-condensed', 'table-hover']
        ModelName = Col('Назва моделі')
        StageData=NestedTableCol("Дані етапу", SubStageTable)
        TimeCol=Col('Час роботи')


