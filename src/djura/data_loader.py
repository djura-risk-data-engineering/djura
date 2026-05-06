"""Download, cache, and load the bundled NGA-West2 pickle dataset.

The dataset is too large (>100 MB) to ship inside the wheel, so it is
hosted as a gzip-compressed asset on a GitHub Release and fetched on
first use into a per-user cache directory.
"""

import gzip
import hashlib
import os
import pickle
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PACKAGE_NAME = "djura"
DATA_FILENAME = "NGA_W2_v2.pickle"

# Update both constants (and re-run the release-data workflow) when the
# dataset changes. Compute the new hash with:
#   python -c "import hashlib,sys; print(hashlib.file_digest(open(sys.argv[1],'rb'),'sha256').hexdigest())" NGA_W2_v2.pickle.gz
GITHUB_RELEASE_URL = (
    "https://github.com/djura-risk-data-engineering/djura/releases/download/"
    "data-v1/NGA_W2_v2.pickle.gz"
)
EXPECTED_SHA256 = (
    # SHA-256 of the compressed .gz asset at the URL above.
    # Fill this in by running the command in the comment above against the
    # actual release asset, then commit the result.
    ""
)

# Refuse downloads larger than 500 MB (uncompressed pickle is ~107 MB).
_MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024


def _cache_dir() -> Path:
    return Path.home() / ".cache" / PACKAGE_NAME


def _cache_path() -> Path:
    return _cache_dir() / DATA_FILENAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_and_extract(dest: Path) -> None:
    url = GITHUB_RELEASE_URL
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_gz = dest.with_suffix(dest.suffix + ".gz.part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, \
                open(tmp_gz, "wb") as out:
            downloaded = 0
            while chunk := response.read(1 << 20):
                downloaded += len(chunk)
                if downloaded > _MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(
                        f"Download from {url} exceeded "
                        f"{_MAX_DOWNLOAD_BYTES // (1024**2)} MB limit — "
                        "aborting."
                    )
                out.write(chunk)
    except urllib.error.HTTPError as e:
        tmp_gz.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download dataset from {url} (HTTP {e.code}). "
            "Make sure the GitHub Release exists and the asset is public."
        ) from e
    except urllib.error.URLError as e:
        tmp_gz.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download dataset from {url}: {e.reason}. "
            "Check your network connection."
        ) from e

    if EXPECTED_SHA256:
        actual = _sha256(tmp_gz)
        if actual != EXPECTED_SHA256:
            tmp_gz.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for downloaded asset.\n"
                f"  expected: {EXPECTED_SHA256}\n"
                f"  actual:   {actual}\n"
                "The file may be corrupted or tampered with. "
                "Delete the partial download and try again, or report the "
                "issue at https://github.com/djura-risk-data-engineering/djura/issues"
            )

    tmp_pkl = dest.with_suffix(dest.suffix + ".part")
    try:
        with gzip.open(tmp_gz, "rb") as gz_in, open(tmp_pkl, "wb") as pkl_out:
            shutil.copyfileobj(gz_in, pkl_out)
        tmp_pkl.replace(dest)
    finally:
        tmp_gz.unlink(missing_ok=True)
        tmp_pkl.unlink(missing_ok=True)


def load_data() -> Any:
    """Return the deserialized dataset, downloading and caching it if needed."""
    cache = _cache_path()
    if not cache.exists():
        _download_and_extract(cache)
    with open(cache, "rb") as f:
        return pickle.load(f)


def clear_cache() -> None:
    """
    Remove the cached dataset so it is re-downloaded on next ``load_data()``.
    """
    global _nga_west2
    _nga_west2 = None
    cache = _cache_path()
    cache.unlink(missing_ok=True)


_nga_west2: Any = None


def get_nga_west2() -> Any:
    """Return the NGA-West2 metadata, loading it at most once per process.

    Override the source by setting the ``DJURA_METADATA_PATH`` environment
    variable to the path of a custom pickle file.
    """
    global _nga_west2
    if _nga_west2 is None:
        custom = os.environ.get("DJURA_METADATA_PATH")
        if custom:
            with open(custom, "rb") as f:
                _nga_west2 = pickle.load(f)
        else:
            _nga_west2 = load_data()
    return _nga_west2
