import ldap

from flask_wtf import FlaskForm
from flask_login import login_user, logout_user
from wtforms import TextField, PasswordField
from wtforms.validators import InputRequired


class User():
  def __init__(self, uid, given_name):
    self.uid = uid
    self.given_name = given_name

  def get_id(self):
    return self.uid

  def is_authenticated(self):
    return True

  def is_active(self):
    return True

  def is_anonymous(self):
    return False


class UserManager():
  users = dict()

  @staticmethod
  def get_user(conn, uid):
    if uid in UserManager.users:
      return UserManager.users[uid]
    else:
      return None

  @staticmethod
  def try_login(conn, username, password):
    conn.simple_bind_s(
      'uid={},cn=users,cn=accounts,dc=kumalab,dc=local'.format(username),
      password)

    ldap_user_object = conn.search_s(
        'uid={},cn=users,cn=accounts,dc=kumalab,dc=local'.format(username),
        ldap.SCOPE_BASE)[0][1]

    user = User(uid=ldap_user_object['uid'][0].decode('UTF-8'),
                given_name=ldap_user_object['givenName'][0].decode('UTF-8'))
    UserManager.users[user.uid] = user

    login_user(user)

    return user

  @staticmethod
  def logout(user):
    if user is not None:
      logout_user()

    if user in UserManager.users:
      del UserManager.users[user]


class LoginForm(FlaskForm):
  username = TextField('Username', [InputRequired()])
  password = PasswordField('Password', [InputRequired()])
