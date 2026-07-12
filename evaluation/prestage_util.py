"""Prestage large files to a local fast disk to avoid slow network-storage loads
and occasional ENOTDIR errors.

On demand: copy only the ckpt the caller actually loads; files under min_mb are returned as-is.
Idempotent: an existing local copy with matching size is reused, not re-copied.
Robust: stat/copy retry (re-listing the parent dir to refresh the FUSE cache) to tolerate
transient ENOTDIR; on total failure, fall back to the original path.
"""
import hashlib
import logging
import os
import shutil
import time

logger = logging.getLogger(__name__)


def _refresh(path: str) -> None:
    try:
        os.listdir(os.path.dirname(path))   # re-list to refresh FUSE metadata cache
    except OSError:
        pass


def prestage(path: str, prestage_dir: str, min_mb: float = 300.0, retries: int = 6) -> str:
    """Copy path to a local copy under prestage_dir and return its path.
    Returns path unchanged if prestage_dir is empty / file missing / smaller than min_mb / copy fails."""
    if not prestage_dir or not path:
        return path

    # stat (with retry)
    st = None
    for i in range(retries):
        try:
            st = os.stat(path)
            break
        except OSError:
            _refresh(path)
            time.sleep(0.5 * (i + 1))
    if st is None:
        return path  # size unavailable; let the caller error out
    if st.st_size < min_mb * 1024 * 1024:
        return path  # small file; not worth prestaging

    os.makedirs(prestage_dir, exist_ok=True)
    key = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:12]
    dst = os.path.join(prestage_dir, f"{key}_{os.path.basename(path)}")
    if os.path.isfile(dst) and os.path.getsize(dst) == st.st_size:
        logger.info(f"[prestage] reusing local copy {dst}")
        return dst  # already staged and complete

    gb = st.st_size / 1e9
    for i in range(retries):
        try:
            t0 = time.time()
            shutil.copyfile(path, dst)
            if os.path.getsize(dst) == st.st_size:
                logger.info(f"[prestage] {path} -> {dst} ({gb:.1f}GB, {time.time()-t0:.0f}s)")
                return dst
        except OSError:
            pass
        _refresh(path)
        time.sleep(1.0 * (i + 1))

    logger.warning(f"[prestage] prestage failed, falling back to original path: {path}")
    try:
        if os.path.exists(dst):
            os.remove(dst)   # clean up partial file
    except OSError:
        pass
    return path
