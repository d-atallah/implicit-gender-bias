from gauth.gdrive_auth import auth

try: gauth
except NameError:
    gauth = auth()