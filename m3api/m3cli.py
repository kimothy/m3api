import typer
import rich
import m3api
import getpass
import sys
import re
from typing import Optional


app = typer.Typer()
state = {"verbose": True}


################################################################################
### Utilities                                                                ###
################################################################################


def cli_endpoint_get(endpoint: str):
    try:
        credentials = m3api.endpoint_get(endpoint)

    except KeyError:
        print(f'Endpoint {endpoint} is not valid')
        sys.exit()

    return credentials


def split_uppercase_arguments(args: list[str]|None) -> dict[str, str]:
    if args is None:
        return {}
    
    pattern = re.compile('[A-Z,0-9]*=.*')
    ok = all(pattern.match(a) for a in args)
    
    if not ok:
        print('All API input arguments must be in the form KEY=VALUE')
        sys.exit()

    mapping = {k: v for k, v in (a.split("=") for a in args)}
    return mapping


################################################################################
### Commands                                                                 ###
################################################################################


@app.command()
def endpoint(delete: Optional[bool] = False):
    key = input('Enter endpoint key: ')

    if key.strip() == '':
        return

    try:
        endpoint = m3api.endpoint_get(key)

    except KeyError:
        endpoint = m3api.Endpoint('', '', '')  #type: ignore
            
    URL = input(f'URL [{endpoint.url}]: ')
    USR = input(f'USR [{endpoint.usr}]: ')
    PWD = getpass.getpass(f'PWD [{"*"*len(endpoint.pwd)}]: ')
            
    URL = URL if URL != '' else endpoint.url
    USR = USR if USR != '' else endpoint.usr
    PWD = PWD if PWD != '' else endpoint.pwd
                    
    m3api.endpoint_set(key, URL, USR, PWD)


@app.command()
def progs(endpoint: str):
    credentials = cli_endpoint_get(endpoint)
    with m3api.MIClient(*credentials) as client:
        progs = client.progs()

    print(progs)

    
@app.command()
def meta(endpoint: str, program: str):
    credentials = cli_endpoint_get(endpoint)
    with m3api.MIClient(*credentials) as client:
        meta = client.meta(program)

    print(meta)


@app.command()
def execute(
        endpoint: str,
        program: str,
        transaction: str,
        tabular: Optional[bool] = False,
        parameters: Optional[list[str]] = typer.Argument(None)):
    
    credentials = cli_endpoint_get(endpoint)
    kwargs = (split_uppercase_arguments(parameters))

    with m3api.MIClient(*credentials) as client:
        try:
            result = client.execute(program, transaction, **kwargs)

        except Exception as error:
            print(error)
            sys.exit()

    print(result)


################################################################################
### Main                                                                     ###
################################################################################


if __name__ == "__main__":
    app()
