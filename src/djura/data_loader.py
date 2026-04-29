"""Download, cache, and load the bundled NGA-West2 pickle dataset.

The dataset is too large (>100 MB) to ship inside the wheel, so it is
hosted as a gzip-compressed asset on a GitHub Release and fetched on
first use into a per-user cache directory.
"""

import gzip
import pickle
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PACKAGE_NAME = "djura"
DATA_FILENAME = "NGA_W2_v2.pickle"

# Update VERSION (and re-run the release-data workflow) when the dataset
# changes.
GITHUB_RELEASE_URL = (
    "https://github.com/djura-risk-data-engineering/djura/releases/download/"
    "data-v1/NGA_W2_v2.pickle.gz"
)


def _cache_dir() -> Path:
    return Path.home() / ".cache" / PACKAGE_NAME


def _cache_path() -> Path:
    return _cache_dir() / DATA_FILENAME


def _download_and_extract(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_gz = dest.with_suffix(dest.suffix + ".gz.part")
    try:
        with urllib.request.urlopen(url) as response, open(tmp_gz, "wb") \
                as out:
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as e:
        tmp_gz.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download dataset from {url} (HTTP {e.code}). "
            f"Make sure the GitHub Release exists and the asset is public."
        ) from e
    except urllib.error.URLError as e:
        tmp_gz.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download dataset from {url}: {e.reason}. "
            f"Check your network connection."
        ) from e

    tmp_pkl = dest.with_suffix(dest.suffix + ".part")
    try:
        with gzip.open(tmp_gz, "rb") as gz_in, open(tmp_pkl, "wb") as pkl_out:
            shutil.copyfileobj(gz_in, pkl_out)
        tmp_pkl.replace(dest)
    finally:
        tmp_gz.unlink(missing_ok=True)
        tmp_pkl.unlink(missing_ok=True)


def load_data(url: str = GITHUB_RELEASE_URL) -> Any:
    """
    Return the deserialized dataset, downloading and caching it if needed.
    """
    cache = _cache_path()
    if not cache.exists():
        _download_and_extract(url, cache)
    with open(cache, "rb") as f:
        return pickle.load(f)


def clear_cache() -> None:
    """
    Remove the cached dataset so it is re-downloaded on next ``load_data()``.
    """
    cache = _cache_path()
    cache.unlink(missing_ok=True)
