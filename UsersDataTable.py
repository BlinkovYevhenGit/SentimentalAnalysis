from flask_table import Table, Col, BoolCol, ButtonCol


class UsersTable(Table):
    classes = ['table', 'table-striped', 'table-condensed', 'table-hover']
    _id=Col("Номер користувача,№")
    Login=Col("Номер користувача,№")
    Password=Col("Номер користувача,№")
    Access=BoolCol("Доступ до моделей",yes_display='Так', no_display='Ні')
    Action=ButtonCol("Змінити доступ", "change", url_kwargs=dict(id='_id'))

def makeUserTable(userList):
    userTable=UsersTable(userList)
    return userTable