from dataclasses import dataclass, field, make_dataclass
from httpx import AsyncClient as AsyncSessionX
from requests import Session
from requests.auth import HTTPBasicAuth
from typing import Any, AsyncGenerator, Callable, Iterable, Literal, Optional, TypedDict, Self
from zeep import AsyncClient, Client, Settings
from zeep.transports import Transport, AsyncTransport
from zeep.cache import SqliteCache

import asyncio
import httpx
import keyring
import argparse
import getpass


################################################################################
### Protocols                                                                ###
################################################################################


class MIField(TypedDict):
    name: str
    type: Literal['A', 'D', 'N']
    length: int
    description: str


class MIMetadata(TypedDict):
    Field: list[MIField]


class MINameValue(TypedDict):
    Name: str
    Value: str


class MIRecord(TypedDict):
    RowIndex: int
    NameValue: list[MINameValue]


class MIResult(TypedDict):
    Program: str
    Transaction: str
    Metadata: MIMetadata
    MIRecord: list[MIRecord]


class CMINameValue(TypedDict):
    Name: str
    Value: float|int|str

    
class CMIRecord(TypedDict):
    RowIndex: int
    NameValue: list[CMINameValue]

    
class CMIResult(TypedDict):
    Program: str
    Transaction: str
    Metadata: MIMetadata
    MIRecord: list[CMIRecord]


################################################################################
### Exceptions                                                               ###
################################################################################


class CoerceError(ValueError):
    pass


################################################################################
### Endpoints                                                                ###
################################################################################


class Endpoint(tuple):
    def __new__(cls, url: str, usr: str, pwd: str):
        return tuple.__new__(Endpoint, (url, usr, pwd))
        
    @property
    def url(self):
        return self[0]

    @property
    def usr(self):
        return self[1]
    
    @property
    def pwd(self):
        return self[2]

    def __repr__(self) -> str:
        return f'Endpoint({self.url}, {self.usr}, {"*" * len(self.pwd)})'


def endpoint_get(key: str) -> Endpoint:
    secret = keyring.get_password('m3api', key)

    if secret is None:
        raise KeyError(f'No endpoints saved with key "{key}"!')
    
    url = secret.rsplit('@', 1)[1]
    usr = secret.rsplit('@', 1)[0].split(':', 1)[0]
    pwd = secret.rsplit('@', 1)[0].split(':', 1)[1]
    return Endpoint(url, usr, pwd)


def endpoint_set(key: str, url: str, usr: str, pwd: str):
    keyring.set_password('m3api', key, f'{usr}:{pwd}@{url}')


def endpoint_del(key: str):
    keyring.delete_password('m3api', key)


################################################################################
### Helper functions                                                         ###
################################################################################


def get_type(field: MIField) -> type:
    match field['type']:
        case 'A':
            return str

        case 'D':
            return int

        case 'N':
            return type(float|int)

        case _:
            raise ValueError(f'Field type {field["type"]} not in ["A", "D","N"]')


def mapparams(**kwargs):
    """Maps the input parameters to a dictionary which can be sent to the clients for
    execution without further modifcation. lowecase keyword arguments are treated as
    a shorthand for known API inputs. Uppercase keyword arguments are treated as input
    parameters, and will be sent directly to the API as such. Camel case keywords are
    sent through to the API as is, without modication.

    Known lower case arguments are:
        cono: int
        divi: int
        maxrecs: int
        program: str
        transaction: str
        returncols: str
        returnmeta: bool|None
        returnempty: bool

    A ValueError is raised if unknown lower case keyword arguments are passed."""
    
    lowercase = {k: v for k, v in kwargs.items() if k.islower()}
    uppercase = {k: v for k, v in kwargs.items() if k.isupper()}
    camelcase = kwargs
    
    parameters = {}

    if 'cono' in kwargs:
        cono = kwargs.pop('cono')
        parameters['Cono'] = cono

    if 'divi' in kwargs:
        divi = kwargs.pop('divi')
        parameters['Divi'] = divi

    if 'maxrecs' in kwargs:
        maxrecs = kwargs.pop('maxrecs')
        parameters['MaxReturnedRecords'] = int(maxrecs)
    
    if 'program' in kwargs:
        program = kwargs.pop('program')
        parameters['Program'] = program

    if 'transaction' in kwargs:
        transaction = kwargs.pop('transaction')
        parameters['Transaction'] = transaction

    if 'returncols' in kwargs:
        returncols = {'ColumnName': kwargs.pop('returncols').split(',')}
        parameters['ReturnColumns'] = returncols

    if 'returnmeta' in kwargs:
        returnmeta = {True: 'MetadataOnly', None: 'MetadataIncluded', False: 'MetadataExcluded'}[kwargs.pop('returnmeta')]
        parameters['ReturnMetadata'] = returnmeta
    
    if 'returnempty' in kwargs:
        returnempthy = kwargs.pop('returnempty')
        parameters['ExcludeEmptyValues'] = not returnempthy

    if any([key.islower() for key in kwargs.keys()]):
        raise ValueError(f'Unknown lower case parameters passed!: {lowercase}')

    mirecord = [{'RowIndex': 0, 'NameValue': [{'Name': k, 'Value': v} for k, v in uppercase.items()]}]
    parameters['MIRecord'] = mirecord

    if 'ExcludeEmptyValues' not in parameters:
        parameters['ExcludeEmptyValues'] = False
    
    parameters.update(camelcase)

    return parameters


################################################################################
### Clients                                                                  ###
################################################################################


cache = SqliteCache()


class MIClient:    
    def __init__(self, url, usr, pwd):
        wsdl = f'{url}/m3api/MIAccess?wsdl'
        
        session = Session()
        session.auth = HTTPBasicAuth(usr, pwd)

        settings = Settings(strict=True)
        transport = Transport(session=session, cache=cache)
        client = Client(wsdl=wsdl, settings=settings, transport=transport)

        self.session = session
        self.client = client

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()

    def close(self):
        self.session.close()

    def execute(self, program: str, transaction: str, **kwargs) -> MIResult:
        """Sends a call to the API and returns the result as a dictionary.

        Upper case keyword arguments are considered input parameters.
        Lower case keyword argumetns are considered known api parameters.
        Camel case keyword arguments are passed on to the api directly.

        Known lower case arguments are:
            cono: int
            divi: int
            maxrecs: int
            program: str
            transaction: str
            returncols: str
            returnmeta: bool|None
            returnempty: bool
        """
        parameters = mapparams(program=program, transaction=transaction, **kwargs)
        data = self.client.service['Execute'](parameters)
        return data

    def progs(self):
        return self.client.service.ListPrograms()

    def meta(self, program: str):
        return self.client.service.GetMetaData(program)


class AsyncMIClient:
    def __init__(self, url, usr, pwd, semaphore=5, retry=False):
        """Sends a call to the API and returns the result as a dictionary.

        Upper case keyword arguments are considered input parameters.
        Lower case keyword arguments are considered known api parameters.
        Camel case keyword arguments are passed on to the api directly.

        Known lower case arguments are:
            cono: int
            divi: int
            maxrecs: int
            program: str
            transaction: str
            returncols: str
            returnmeta: bool|None
            returnempty: bool

        The semaphore definens how many requests the API is working on
        asynchronously. If retry is set to True, the request will be
        performed an additional time in case of a connection error,
        or a timeout exception. If it fails again, the Exception will
        be raised."""

        session = AsyncSessionX(auth=(usr, pwd))
        
        wsdl = f'{url}/m3api/MIAccess?wsdl'
        settings = Settings(strict=True)
        transport = AsyncTransport(client=self.session, cache=cache)
        client = AsyncClient(wsdl=wsdl, settings=settings, transport=transport)

        self.retry = retry
        self.client = client
        self.session = session
        self.semaphore = asyncio.Semaphore(semaphore)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self.session.is_closed:
            await self.session.aclose()

    def close(self):
        if not self.session.is_closed:
            self.session.aclose()

    async def execute(self, program: str, transaction: str, **kwargs) -> MIResult:
        params = mapparams(program=program, transaction=transaction, **kwargs)

        async with self.semaphore:
            try:
                result = await self.client.service['Execute'](params)

            except (httpx.TimeoutException, httpx.ConnectError) as error:
                if self.retry:
                    await asyncio.sleep(1)
                    result =  await self.execute(**params)
                
                else:
                    raise error
        
        return result

    async def progs(self):
        return await self.client.service.ListPrograms()

    async def meta(self, program: str):
        return await self.client.service.GetMetaData(program)


class MPDClient:
    def __init__(self, url, usr, pwd, service: str):
        self.auth = HTTPBasicAuth(usr, pwd)
        self.session = Session()
        self.session.auth = self.auth
        self.service = service
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
        self.session = None

    def client(self, url: str) -> Client:
        return Client(url, transport=Transport(session=self.session))

    def close(self):
        self.session.close()

    def execute(self, service: str, operation: str, **kwargs):
        client = self.client(f"{self.url}/mws-ws/services/{service}?wsdl")
        return getattr(client.service, operation)(kwargs)

    def services(self):
        client = self.client(f"{self.url}/mws-ws/core/Discovery?wsdl")
        return client.service.GetServices()


class AsyncMPDClient:
    def __init__(self, url, usr, pwd, service: str, semaphore=5, retry=False):
        self.service: str = service
        self.session: AsyncSessionX = AsyncSessionX(auth=(usr, pwd))
        self.client: AsyncClient = AsyncClient(
            wsdl=f"{url}/mws-ws/services/{service}?wsdl",
            settings=Settings(strict=True),
            transport=AsyncTransport(client=self.session))

        self.retry = retry
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(semaphore)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self.session.is_closed:
            await self.session.aclose()

    def close(self):
        if not self.session.is_closed:
            self.session.aclose()

    async def execute(self, operation: str, **kwargs):
        async with self.semaphore:            
            try:
                return await self.client.service[operation](kwargs)

            except (httpx.TimeoutException, httpx.ConnectError) as error:
                if self.retry:
                    await asyncio.sleep(1)
                    return await self.execute(operation, **kwargs)
                

class ClientFactory:
    def __init__(self, url: str, usr: str, pwd: str, retry=False):
        self.endpoint = Endpoint(url, usr, pwd)
        self.retry = retry
        
    @classmethod
    def from_endpoint(cls, key: str) -> Self:
        return cls(*endpoint_get(key))

    def async_mi(self, semaphore: Optional[int] = 5) -> AsyncMIClient:
        return AsyncMIClient(*self.endpoint, semaphore, self.retry) #type: ignore

    def async_mpd(self, service, semaphore: Optional[int] = 5) -> AsyncMPDClient:
        return AsyncMPDClient(*self.endpoint, service, semaphore, self.retry) #type: ignore

    def mi(self) -> MIClient:
        return MIClient(*self.endpoint)

    def mpd(self) -> MPDClient:
        return MPDClient(*self.endpoint)


################################################################################
### Converter Constructers                                                   ###
################################################################################


def construct_coerce_field(field: MIField) -> Callable[[MINameValue], CMINameValue]:
    match field['type']:
        case 'D':
            def converter(value):
                return int(value.strip())
                
        case 'A':
            def converter(value):
                return value.strip()

        case 'N':
            def converter(value):
                try:
                    
                    return int(value.strip())

                except ValueError:
                    pass
            
                try:
                    return float(value.strip())

                except ValueError:
                    pass

                if value.strip() == '':
                    return float('NaN')

                else:
                    raise CoerceError(f'''Could not coerce "{value}" with fieldtype "{field['type']}" to an int or float''')

        case _:
            raise CoerceError(f'''No match for fieldtype "{field['type']}"''')
        
        
    def coerce_field(name_value: MINameValue) -> CMINameValue:
        f'''Coerce field "{field['name']}" for type "{field['type']}".'''
        name = name_value['Name']
        value = converter(name_value['Value'])
        return CMINameValue(Name=name, Value=value)

    return coerce_field


def construct_coerce_record(metadata: MIMetadata) -> Callable[[MIRecord], CMIRecord]:
    converters = tuple(construct_coerce_field(f) for f in metadata['Field'])

    def coerce_record(record: MIRecord) -> CMIRecord:
        row_index = record['RowIndex']
        name_value = [c(f) for c, f in zip(converters, record['NameValue'])]
        cmi_record = CMIRecord(RowIndex=row_index, NameValue=name_value)

        return cmi_record

    return coerce_record


def construct_coerce_result(metadata: MIMetadata) -> Callable[[MIResult], CMIResult]:
    converter = construct_coerce_record(metadata)
    
    def coerce_result(result: MIResult) -> CMIResult:
        mi_record = [converter(r) for r in result['MIRecord']]
        program = result['Program']
        transaction = result['Transaction']
        metadata = result['Metadata']
        cmi_result = CMIResult(Program=program, Transaction=transaction, Metadata=metadata, MIRecord=mi_record)

        return cmi_result

    return coerce_result


def construct_coerce_dict(metadata: MIMetadata) -> Callable[[MIResult], Iterable[dict]]:
    converter = construct_coerce_record(metadata)
    
    def coerce_result(result: MIResult) -> Iterable[dict]:
        for r in result['MIRecord']:
            record = {f['Name']: f['Value'] for f in converter(r)['NameValue']}
            yield record

    return coerce_result


def construct_raw_dict(metadata: MIMetadata) -> Callable[[MIResult], Iterable[dict]]:
    def result(result: MIResult) -> Iterable[dict]:
        for r in result['MIRecord']:
            record = {f['Name']: f['Value'] for f in r['NameValue']}
            yield record

    return result


def construct_dclass(metadata: MIMetadata, program: str, transaction: str, frozen=True):
    cls_name = f"{program}_{transaction}"
    mi_field = metadata['Field']
    fields = [(f['name'], get_type(f), field(metadata=f)) for f in mi_field]
    
    dclass = make_dataclass(
        cls_name=cls_name,
        fields=fields,
        order=frozen,
        frozen=frozen,
        slots=frozen)

    return dclass


################################################################################
### Result Converters                                                        ###
################################################################################


def to_dict(mi_result: MIResult, coerce=True) -> Iterable[dict]:
    """Converts a MIResult dict to an iterable of dictionary records.
    If the parameter coerce is True, the values will be stripped and
    coerced."""
    if coerce:
        converter = construct_coerce_dict(mi_result['Metadata'])

    else:
        converter = construct_raw_dict(mi_result['Metadata'])

    yield from converter(mi_result)


def to_dclass(mi_result: MIResult, frozen=True) -> Iterable[Any]:
    """Converts a MIResult dictionary to an iterable of imutable
    dataclasses. Metadata is stored in the dataclass as type hints, as
    well as field metadata under the key MIMetadata.

    The frozen parameter makes the dataclass ordered, imutable, and
    turns on slots. This is faster and safer to use.
    """

    metadata = mi_result['Metadata']
    program = mi_result['Program']
    transaction = mi_result['Transaction']
    dclass = construct_dclass(metadata, program, transaction)
    
    for r in to_dict(mi_result, coerce=True):
        yield dclass(**r)



################################################################################
### CLI                                                                      ###
################################################################################


def cli():
    key = input('Enter endpoint key: ')

    if key.strip() == '':
        return

    try:
        endpoint = endpoint_get(key)

    except KeyError:
        endpoint = Endpoint('', '', '')
            
        try:
            URL = input(f'URL [{endpoint.url}]: ')
            USR = input(f'USR [{endpoint.usr}]: ')
            PWD = getpass.getpass(f'PWD [{"*"*len(endpoint.pwd)}]: ')
            
            URL = URL if URL != '' else endpoint.url
            USR = USR if USR != '' else endpoint.usr
            PWD = PWD if PWD != '' else endpoint.pwd
                    
        except Exception as error:
            raise error

        else:
            endpoint_set(key, URL, USR, PWD)
            print(f'Endpoint "{key}" saved!')


################################################################################
### Main                                                                     ###
################################################################################


if __name__ == '__main__':
    cli()
