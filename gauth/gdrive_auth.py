def auth():
    from pydrive2.auth import GoogleAuth
    settings = {
        "client_config_backend": "file",
        "client_config_file": "gauth/client_secrets.json",
        }
    gauth = GoogleAuth(settings = settings)
    gauth.LocalWebserverAuth()
    # .ServiceAuth() 
    return gauth