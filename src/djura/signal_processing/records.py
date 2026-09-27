# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026 Djura | Risk - Data - Engineering S.r.l.
"""Record readers."""
import zipfile
import difflib
import numpy as np
from datetime import datetime


def content_from_zip(paths, zip_name):
    """
    Details
    -------
    This function reads the contents of all selected records
    from the zipfile in which the records are located

    Parameters
    ----------
    paths : list
        Containing file list which are going to be read from the zipfile.
    zip_name : str
        Path to the zip file where file lists defined in "paths" are located.

    Returns
    -------
    contents : dictionary
        Containing raw contents of the files which are read from the zipfile.
    """

    contents = {}
    with zipfile.ZipFile(zip_name, 'r') as myzip:
        for i in range(len(paths)):
            with myzip.open(paths[i]) as myfile:
                contents[i] = [x.decode('utf-8') for x in myfile.readlines()]

    return contents


def read_nga(in_filename=None, content=None, out_filename=None):
    """
    Details
    -------
    This function process acceleration history for NGA data file (.AT2 format).

    Parameters
    ----------
    in_filename : str, optional
        Location and name of the input file.
        The default is None
    content : str, optional
        Raw content of the .AT2 file.
        The default is None
    out_filename : str, optional
        location and name of the output file.
        The default is None.

    Notes
    -----
    At least one of the two variables must be defined: inFilename, content.

    Returns
    -------
    dt : float
        time interval of recorded points.
    npts : int
        number of points in ground motion record file.
    desc : str
        Description of the earthquake (e.g., name, year, etc).
    t : numpy.ndarray (n x 1)
        time array, same length with npts.
    acc : numpy.ndarray (n x 1)
        acceleration array, same length with time unit
        usually in (g) unless stated as other.
    """

    try:
        # Read the file content from inFilename
        if content is None:
            with open(in_filename, 'r') as inFileID:
                content = inFileID.readlines()

        # check the first line
        temp = str(content[0]).split()
        try:  # description is in the end
            # do a test with str to float conversion,
            # this will be ok if description is in the end.
            float(temp[0])
            # Description of the record
            desc = content[-2]
            # Number of points and time step of the record
            row4Val = content[-4]
            # Acceleration values
            acc_data = content[:-4]
        except ValueError:  # description is in the beginning
            # Description of the record
            desc = content[1]
            # Number of points and time step of the record
            row4Val = content[3]
            # Acceleration values
            acc_data = content[4:]

        # Description of the record
        desc = desc.replace('\r', '')
        desc = desc.replace('\n', '')
        # Number of points and time step of the record
        if row4Val[0][0] == 'N':
            val = row4Val.split()
            if 'dt=' in row4Val:
                dt_str = 'dt='
            elif 'DT=' in row4Val:
                dt_str = 'DT='
            if 'npts=' in row4Val:
                npts_str = 'npts='
            elif 'NPTS=' in row4Val:
                npts_str = 'NPTS='
            if 'sec' in row4Val:
                sec_str = 'sec'
            elif 'SEC' in row4Val:
                sec_str = 'SEC'
            npts = int(val[(val.index(npts_str)) + 1].rstrip(','))
            try:
                dt = float(val[(val.index(dt_str)) + 1])
            except ValueError:
                dt = float(val[(val.index(dt_str)) + 1].replace(
                    sec_str + ',', ''))
        else:
            val = row4Val.split()
            npts = int(val[0])
            dt = float(val[1])

        # Acceleration values
        acc = np.array([])
        for line in acc_data:
            acc = np.append(acc, np.array(line.split(), dtype=float))
        dur = len(acc) * dt
        t = np.arange(0, dur, dt)

        if out_filename is not None:
            np.savetxt(out_filename, acc, fmt='%1.4e')

        return dt, npts, desc, t, acc

    except BaseException as error:
        print(f"Record file reader FAILED for {in_filename}: ", error)


def read_esm(in_filename=None, content=None, out_filename=None):
    """
    Details
    -------
    This function process acceleration history for ESM data file.

    Parameters
    ----------
    in_filename : str, optional
        Location and name of the input file.
        The default is None
    content : str, optional
        Raw content of the ESM record file.
        The default is None
    out_filename : str, optional
        location and name of the output file.
        The default is None.

    Returns
    -------
    dt : float
        time interval of recorded points.
    npts : int
        number of points in ground motion record file.
    desc : str
        Description of the earthquake (e.g., name, year, etc).
    time : numpy.ndarray (n x 1)
        time array, same length with npts.
    acc : numpy.ndarray (n x 1)
        acceleration array, same length with time unit
        usually in (g) unless stated as other.
    """

    try:
        # Read the file content from inFilename
        if content is None:
            with open(in_filename, 'r') as inFileID:
                content = inFileID.readlines()

        desc = content[:64]
        dt = float(difflib.get_close_matches(
            'SAMPLING_INTERVAL_S', content)[0].split()[1])
        npts = len(content[64:])
        acc_data = content[64:]
        acc = np.asarray([float(data) for data in acc_data], dtype=float)
        dur = len(acc) * dt
        t = np.arange(0, dur, dt)
        acc = acc / 980.655  # cm/s**2 to g

        if out_filename is not None:
            np.savetxt(out_filename, acc, fmt='%1.4e')

        return dt, npts, desc, t, acc

    except BaseException as error:
        print(f"Record file reader FAILED for {in_filename}: ", error)


def convert_esm_to_peer_format(
    in_filename=None, content=None, out_filename=None, **kwargs
):
    """
    Converts an ESM format ground motion record to PEER .AT2 format.

    Parameters
    ----------
    in_filename : str, optional
        Path to save the input ESM-style file.
        The default is None
    content : str, optional
        Raw content of the ESM record file.
        The default is None
    out_filename : str, optional
        Path to save the converted PEER-style file.
        The default is None.

    Returns
    -------
    dt : float
        time interval of recorded points.
    acc : numpy.ndarray (n x 1)
        acceleration array, same length with time unit
        usually in (g) unless stated as other.
    """
    dt, npts, header_lines, _, acc = read_esm(
        in_filename=in_filename, content=content)

    # Extract metadata from header
    metadata = {}
    for line in header_lines:
        if ':' in line:
            key, value = line.strip().split(":", 1)
            metadata[key.strip()] = value.strip()

    # Get sampling interval, number of points, and event description
    dt = float(metadata.get("SAMPLING_INTERVAL_S"))
    event = metadata.get("EVENT_NAME", "UNKNOWN EVENT")
    if event == 'None':
        event = "UNKNOWN EVENT"
    if "event" in kwargs.keys():
        event = kwargs["event"]
    comp = metadata.get("STREAM")
    station = f"{metadata.get('NETWORK')}-{metadata.get('STATION_CODE')}"

    date_str = metadata.get('EVENT_DATE_YYYYMMDD', 'UNKNOWN_DATE')
    if date_str != 'UNKNOWN_DATE':
        date = datetime.strptime(date_str, "%Y%m%d").strftime("%m/%d/%Y")

    # Build PEER-style header
    line1 = "ESM STRONG MOTION DATABASE RECORD\n"
    line2 = f"{event}, {date}, {station}, {comp}\n"
    line3 = "ACCELERATION TIME SERIES IN UNITS OF G\n"
    line4 = f"NPTS= {npts:6d}, DT= {dt:7.4f} SEC,\n"

    # Format acceleration values (5 per line, scientific notation)
    data_str_lines = []
    for i in range(0, len(acc), 5):
        chunk = acc[i:i + 5]
        line = "  ".join([f"{val: .7E}" for val in chunk])
        data_str_lines.append(line + "\n")

    # Combine all parts and write to file
    full_content = [line1, line2, line3, line4] + data_str_lines

    with open(out_filename, 'w') as f:
        f.writelines(full_content)

    return dt, acc
