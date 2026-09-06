# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Tuple, Union
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED

import requests


logger = logging.getLogger(__name__)


class DownloaderBase(ABC):
    """
    Abstract Base Class for Downloaders
    """
    ZIPFILE_NAME: str = 'UnscaledRecords.zip'
    """Name of the zipfile which contains the downloaded records."""
    FAILURES_NAME: str = 'FailedRecords.txt'
    """Name of the report listing the records that could not be downloaded."""

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

    Examples
    --------
    >>> downloader = ESMDownloader(          # doctest: +SKIP
    ...     download_dir=".",
    ...     events=["EMSC-20161030_0000029"],
    ...     stations=["T1213"],
    ...     username="you@example.com",
    ...     password="secret",
    ... )
    >>> downloader.download()                # doctest: +SKIP
    """

    TOKEN_URL: str = 'https://esm-db.eu/esmws/generate-signed-message/1/query'
    """Endpoint issuing the signed message used to authenticate downloads."""
    EVENTDATA_URL: str = 'https://esm-db.eu/esmws/eventdata/1/query'
    """Endpoint serving the record waveforms."""
    TIMEOUT: int = 120
    """Seconds to wait for a response before giving up on a request."""

    token_path: Path
    """Download path for token used to retrieve records from ESM database."""
    failed_records: List[Tuple[str, str, str]]
    """Records the last :meth:`download` could not retrieve, as
    ``(event, station, reason)`` triples. Empty until it has run."""
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
            Path to the download directory. Created if it does not exist.
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
            Download path for token used to retrieve records from ESM
            database. An existing file there is reused as the token;
            otherwise the token fetched with the credentials is written
            to it. By default "", i.e. ``download_dir / "token.txt"``.

        Raises
        ------
        ValueError
            ``events`` and ``stations`` differ in length, since the two are
            paired element-wise into one request per record.
        """

        if len(events) != len(stations):
            raise ValueError(
                f"events and stations must be paired element-wise, but got "
                f"{len(events)} event(s) and {len(stations)} station(s).")

        self.download_dir = Path(download_dir)
        self.token_path = (
            Path(token_path) if token_path
            else self.download_dir / 'token.txt')
        self.username = username
        self.password = password
        self.events = events
        self.stations = stations
        self.failed_records = []

    def download(self) -> Path:
        """
        Downloads the records with requested station and event IDs.

        Every requested record is attempted: one the database will not
        serve is skipped instead of abandoning the request, so a failure
        partway through still returns the records that did arrive. See
        :meth:`_send_request` for what is reported about the ones left
        out.

        Returns
        -------
        Path
            Path to the zipfile containing downloaded records. It is
            empty when no record could be downloaded at all.

        Raises
        ------
        TypeError
            Neither username and password nor an existing token file are
            provided.
        requests.exceptions.HTTPError
            The ESM database rejected the credentials while issuing a
            token. Failures of the record requests themselves are
            reported rather than raised.
        """

        logger.info('Started executing download method to retrieve selected '
                    'records from https://esm-db.eu')
        self.download_dir.mkdir(parents=True, exist_ok=True)
        if not self.token_path.is_file():
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

        A record the database refuses, or whose request errors out, is
        skipped so that the remaining ones are still downloaded. Those
        left out are collected in :attr:`failed_records` and written to
        ``download_dir / FAILURES_NAME``; the report of an earlier run is
        removed, so it always describes the latest one.

        Returns
        -------
        Path
            Path to the zipfile which contains downloaded records. The
            zipfile is written even when every record failed, in which
            case it is empty and the report names them all.

        Notes
        -----
        The web service serves one record per request, so the records are
        downloaded one by one and collected into a single zipfile.
        """

        logger.info('Sending download requests...')
        # Set the zip path which contains the downloaded records
        zip_path = self.download_dir / self.ZIPFILE_NAME
        # Constant request parameters
        DATA_TYPE = 'ACC'
        FORMAT = 'ascii'
        total = len(self.events)
        self.failed_records = []
        # Create a temporary directory to extract downloaded records
        with TemporaryDirectory() as temp_dir:
            temp_zip = Path(temp_dir) / 'temp.zip'
            # Starting to download each record one by one
            for i, (event, station) in enumerate(
                    zip(self.events, self.stations), start=1):
                # Set the parameters
                params = (
                        ('eventid', event),
                        ('data-type', DATA_TYPE),
                        ('station', station),
                        ('format', FORMAT)
                )
                try:
                    # The token is reopened per request: a file object handed
                    # to requests is read to the end and would upload nothing
                    # on the next request.
                    with open(self.token_path, 'rb') as token:
                        response = requests.post(
                            url=self.EVENTDATA_URL, params=params,
                            files={'message': (self.token_path.name, token)},
                            timeout=self.TIMEOUT)
                    # Check the status code to write the downloaded record
                    if response.status_code != 200:
                        raise requests.exceptions.HTTPError(
                            f'HTTP status code {response.status_code}',
                            response=response)
                    # Write the record to a temporary zip
                    with open(temp_zip, 'wb') as zf:
                        zf.write(response.content)
                    # Extract the record
                    with ZipFile(temp_zip, 'r') as zipObj:
                        zipObj.extractall(temp_dir)
                except (requests.exceptions.RequestException, BadZipFile,
                        OSError) as exc:
                    # One unavailable record must not cost the whole suite.
                    reason = f'{type(exc).__name__}: {exc}'
                    self.failed_records.append((event, station, reason))
                    logger.warning('%d/%d failed for event %s at station '
                                   '%s (%s).', i, total, event, station,
                                   reason)
                else:
                    logger.info('%d/%d of requests are done.', i, total)
                finally:
                    temp_zip.unlink(missing_ok=True)

            self._report_failures()

            # Now write all back into the brand new zipfile initially targeted
            with ZipFile(zip_path, 'w', ZIP_DEFLATED) as zipObj:
                for file_path in Path(temp_dir).iterdir():
                    zipObj.write(file_path, file_path.name)

        return zip_path

    def _report_failures(self) -> Path:
        """
        Writes :attr:`failed_records` to ``download_dir / FAILURES_NAME``.

        Returns
        -------
        Path
            Path to the report. It is deleted, rather than left stale from
            a previous run, when every record was downloaded.
        """

        failures_path = self.download_dir / self.FAILURES_NAME
        if not self.failed_records:
            failures_path.unlink(missing_ok=True)
            return failures_path

        lines = [
            f'{len(self.failed_records)} of {len(self.events)} record(s) '
            f'could not be downloaded from {self.EVENTDATA_URL}',
            '',
            'event\tstation\treason',
        ]
        lines += [f'{event}\t{station}\t{reason}'
                  for event, station, reason in self.failed_records]
        failures_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        logger.warning('%d of %d record(s) failed; wrote the list to %s',
                       len(self.failed_records), len(self.events),
                       failures_path)
        return failures_path

    def get_token(self) -> None:
        """
        Retrieves the ESM token and writes it to :attr:`token_path`.

        Raises
        ------
        requests.exceptions.HTTPError
            The credentials were rejected by the ESM database.

        Notes
        -------
        The token is the signed message documented at
        https://esm-db.eu/esmws/generate-signed-message/1/query-options.html
        """

        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        message = (f'{{"user_email": "{self.username}", '
                   f'"user_password": "{self.password}"}}')
        response = requests.post(url=self.TOKEN_URL,
                                 files={'message': (None, message)},
                                 timeout=self.TIMEOUT)
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(
                f'Could not obtain an ESM token (HTTP status code '
                f'{response.status_code}). Make sure that the credentials '
                f'are valid.', response=response)
        self.token_path.write_bytes(response.content)
        logger.info('Token written to %s', self.token_path)
