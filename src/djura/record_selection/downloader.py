from abc import ABC, abstractmethod
from pathlib import Path
import requests
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import List, Union
from zipfile import ZipFile, ZIP_DEFLATED


class DownloaderBase(ABC):
    """
    Abstract Base Class for Downloaders
    """
    ZIPFILE_NAME: str = 'UnscaledRecords.zip'
    """Name of the zipfile which contains the downloaded records."""

    @abstractmethod
    def download(self) -> Path:
        """Download method for Downloaders

        Returns
        -------
        Path
            Path to the zipfile containing downloaded records
        """


class ESMDownloader(DownloaderBase):
    """
    This class has been created as an automated web-tool to
    download unscaled record time histories from European Strong
    Motion (ESM) database (https://esm-db.eu).

    Parameters
    ----------
    RecordDownloader : ABC
        Abstract Base Class used for RecordDownloaders
    """

    token_path: Union[str, Path]
    """Download path for token used to retrieve records from ESM database."""
    username: str
    """Account username"""
    password: str
    """Account password"""
    download_dir: Path
    """Path to the download directory"""
    stations: List[str]
    """Records' station codes"""
    events: List[str]
    """Records' event IDs."""

    def __init__(self, download_dir: Union[str, Path],
                 events: List[str], stations: List[str],
                 username: str = "", password: str = "",
                 token_path: Union[str, Path] = "") -> None:
        """
        Constructs ESMDownloader object.

        Parameters
        ----------
        download_dir : Union[str, Path]
            Path to the download directory
        events : List[str]
            Records' event IDs
        stations : List[str]
            Records' station codes
        username : str
            Account username (https://esm-db.eu)
            By default ""
        password : str
            Account password (https://esm-db.eu)
            By default ""
        token_path : Union[str, Path], optional
            Download path for token used to retrieve records from ESM database
            By default ""
        """

        self.token_path = token_path
        self.username = username
        self.password = password
        self.dowload_directory = Path(download_dir)
        self.events = events
        self.stations = stations

    def download(self) -> Path:
        """
        Downloads the records with requested station and event IDs.

        Returns
        -------
        Path
            Path to the zipfile containing downloaded records.

        Raises
        ------
        TypeError
            Neither username and password nor token_path are provided.
        """

        print('\nStarted executing download method to retrieve selected '
              'records from https://esm-db.eu')
        if not self.token_path:
            # Check if username and password are entered
            if self.username and self.password:
                self.get_token()  # Get a new token with credentials
            else:
                raise TypeError(
                    'You have to enter either credentials or path to the token'
                    ' to download records from ESM database.')
        # Send download request and return the downloaded zipfile path
        return self._send_request()

    def _send_request(self) -> Path:
        """
        Sends request to download the records and if successful,
        writes them to ZIPFILE.

        Returns
        -------
        Path
            Path to the zipfile which contains downloaded records.

        Notes
        -----
        Couldn't figure out why but need to download each record
        and zip them into a file.
        """

        print("Sending download requests...")
        # Set the zip path which contains the downloaded records
        zip_path = self.dowload_directory / self.ZIPFILE_NAME
        # Set the request url
        URL = 'https://esm-db.eu/esmws/eventdata/1/query'
        # Set the files used in request
        files = {
                 'message': ('path/to/token.txt', open(self.token_path, 'rb'))
        }
        # Constant request parameters
        DATA_TYPE = 'ACC'
        FORMAT = 'ascii'
        # Create a temporary directory to extract downloaded records
        with TemporaryDirectory() as temp_dir:
            temp_zip = Path(temp_dir) / 'temp.zip'
            # Starting to download each record one by one
            for i in range(len(self.events)):
                # Set the parameters
                params = (
                        ('eventid', self.events[i]),
                        ('data-type', DATA_TYPE),
                        ('station', self.stations[i]),
                        ('format', FORMAT)
                )
                # Get the response
                response = requests.post(url=URL, params=params, files=files)
                # Check the status code to write the downloaded record
                if response.status_code == 200:
                    # Write the record to a temporary zip
                    with open(temp_zip, 'wb') as zf:
                        zf.write(response.content)
                    # Extract the record
                    with ZipFile(temp_zip, 'r') as zipObj:
                        zipObj.extractall(temp_dir)
                    temp_zip.unlink()  # Delete the temporary file
                    print(f'{i+1}/{len(self.events)} of requests are done.')

                # Something went wrong
                else:
                    print('Problem with the download operation occurred. \n'
                          'Make sure that the credentials or token are valid.')
                    raise requests.exceptions.HTTPError(
                        f"Unexpected HTTP status code: {response.status_code}")
            # Now write all back into the brand new zipfile initially targeted
            with ZipFile(zip_path, 'w', ZIP_DEFLATED) as zipObj:
                for file_path in Path(temp_dir).iterdir():
                    zipObj.write(file_path, file_path.name)

        return zip_path

    def get_token(self) -> None:
        """
        Retrieves the ESM token.

        Notes
        -------
        Data is obtained using any program supporting the
        HTTP-POST method, e.g., CURL. See:
        https://esm-db.eu/esmws/generate-signed-message/1/query-options.html
        """

        if self.token_path:
            self.token_path = Path(self.token_path)
        else:
            self.token_path = self.dowload_directory / 'token.txt'
        # Arguments used to utilise HTTP-POST method via CURL
        args = [
            'curl',
            '-X', 'POST',
            '-F', f'message={{"user_email": "{self.username}", \
                             "user_password": "{self.password}"}}',
            '-o', f'{self.token_path}',
            'https://esm-db.eu/esmws/generate-signed-message/1/query'
        ]
        # In the case of Windows OS, disable revocation checking
        if sys.platform.startswith('win'):
            args.insert(args.index('curl') + 1, '--ssl-no-revoke')
        # Run CURL through the subprocess package
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == '__main__':
    import os
    import pickle
    from dotenv import load_dotenv

    load_dotenv()
    flatfile = Path(__file__).parent / 'assets/flatfile_shallow_v1.pickle'
    with open(flatfile, "rb") as f:
        data = pickle.load(f)

    # Test for ESMDownloader
    username = os.environ["ESM_USERNAME"]
    password = os.environ["ESM_PASSWORD"]
    download_path = Path.cwd()
    station_codes = [s.split('-')[1]
                     for s in data['Station_name'][:2].tolist()]
    event_ids = data['esm_event_id'][:2].tolist()
    downloader = ESMDownloader(download_path, event_ids, station_codes,
                               username, password)
    downloader.download()
