"""__init__.py

This files contains classes that allow easy interaction
with the M3 API's. Both sync and async versions are
provided to allow both simple one liners in scrips,
as well as using asyncio to increase efficiency for
batch uploads etc."""

__author__ = "Kim Timothy Engh"
__copyright__ = "Copyright 2023"
__credits__ = [""]
__license__ = "GPLv3"
__version__ = "0.0.1"
__maintainer__ = "Kim Timothy Engh"
__email__ = "kim.timothy.engh@epiroc.com"
__status__ = "Development"


from . m3api import (
    endpoint_del,
    endpoint_get,
    endpoint_set,
    to_dclass,
    to_dict,
    AsyncMIClient,
    AsyncMPDClient,
    MIClient,
    MPDClient
)
