from __future__ import annotations

import hashlib
import io
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
import pandas as pd
import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def create_session(max_retries: int) -> requests.Session:
    """Create a reusable HTTP session with retry/backoff support."""
    session = requests.Session()

    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=8,
        pool_maxsize=8,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/154.0 Safari/537.36"
            )
        }
    )

    return session


def get_session(max_retries: int) -> requests.Session:
    """Get one HTTP session per worker thread."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = create_session(max_retries)

    return _thread_local.session


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def download_and_validate_image(
    row: tuple,
    image_dir: Path,
    timeout: int,
    max_retries: int,
    jpeg_quality: int,
) -> dict:
    """Download, validate, canonicalize and hash one image."""

    index, sample_id, image_url = row

    output_path = image_dir / f"{sample_id}.jpg"
    session = get_session(max_retries)

    try:
        response = session.get(
            image_url,
            timeout=timeout,
        )

        response.raise_for_status()

        if not response.content:
            raise ValueError("Empty response body.")

        image = Image.open(io.BytesIO(response.content))

        # Verify image integrity before decoding.
        image.verify()

        # Reopen after verify().
        image = Image.open(io.BytesIO(response.content))

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            output_path,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )

        sha256 = compute_sha256(output_path)

        phash = str(imagehash.phash(image))

        return {
            "index": index,
            "id": sample_id,
            "success": True,
            "image_path": str(output_path),
            "sha256": sha256,
            "phash": phash,
            "width": image.width,
            "height": image.height,
            "status_code": response.status_code,
            "content_type": response.headers.get(
                "content-type",
                "",
            ),
            "error": None,
        }

    except (
        requests.RequestException,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        return {
            "index": index,
            "id": sample_id,
            "success": False,
            "image_path": None,
            "sha256": None,
            "phash": None,
            "width": None,
            "height": None,
            "status_code": (response.status_code if "response" in locals() else None),
            "content_type": (
                response.headers.get("content-type", "")
                if "response" in locals()
                else ""
            ),
            "error": str(exc),
        }


def download_images(
    df: pd.DataFrame,
    image_dir: Path,
    timeout: int,
    max_retries: int,
    workers: int,
    jpeg_quality: int,
) -> pd.DataFrame:
    """Download and validate images concurrently."""

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = list(
        zip(
            df.index,
            df["id"],
            df["image_url"],
        )
    )

    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_and_validate_image,
                row,
                image_dir,
                timeout,
                max_retries,
                jpeg_quality,
            )
            for row in rows
        ]

        for future in as_completed(futures):
            results.append(future.result())

    results_df = pd.DataFrame(results)

    return results_df.sort_values("index").reset_index(drop=True)
