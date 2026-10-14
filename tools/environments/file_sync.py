"""Shared file sync manager for remote execution backends.

Tracks local file changes via mtime+size, detects deletions, and
syncs to remote environments transactionally.  Used by SSH, Modal,
and Daytona.  Docker and Singularity use bind mounts (live host FS
view) and don't need this.
"""

import hashlib
import logging
import ntpath
import os
import shlex
import shutil
import signal
import stat
import tarfile
import tempfile
import threading
import time

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows — file locking skipped
from pathlib import Path
from typing import Callable

from hermes_constants import get_hermes_home
from tools.environments.base import _file_mtime_key

logger = logging.getLogger(__name__)

# Keep retry sleeps patchable without mutating the shared stdlib ``time``
# module. Patching ``tools.environments.file_sync.time.sleep`` replaces
# ``time.sleep`` globally because ``time`` is the module object; under xdist
# that lets unrelated background threads inflate retry-test call counts.
_sleep = time.sleep

_SYNC_INTERVAL_SECONDS = 5.0
_FORCE_SYNC_ENV = "HERMES_FORCE_FILE_SYNC"

# Transport callbacks provided by each backend
UploadFn = Callable[[str, str], None]  # (host_path, remote_path) -> raises on failure
BulkUploadFn = Callable[[list[tuple[str, str]]], None]  # [(host_path, remote_path), ...] -> raises on failure
BulkDownloadFn = Callable[[Path], None]  # (dest_tar_path) -> writes tar archive, raises on failure
DeleteFn = Callable[[list[str]], None]  # (remote_paths) -> raises on failure
GetFilesFn = Callable[[], list[tuple[str, str]]]  # () -> [(host_path, remote_path), ...]


def iter_sync_files(container_base: str = "/root/.hermes") -> list[tuple[str, str]]:
    """Enumerate all files that should be synced to a remote environment.

    Combines credentials, skills, and cache into a single flat list of
    (host_path, remote_path) pairs.  Credential paths are remapped from
    the hardcoded /root/.hermes to *container_base* because the remote
    user's home may differ (e.g. /home/daytona, /home/user).
    """
    # Late import: credential_files imports agent modules that create
    # circular dependencies if loaded at file_sync module level.
    from tools.credential_files import (
        get_credential_file_mounts,
        iter_cache_files,
        iter_skills_files,
    )

    files: list[tuple[str, str]] = []
    for entry in get_credential_file_mounts():
        remote = entry["container_path"].replace(
            "/root/.hermes", container_base, 1
        )
        files.append((entry["host_path"], remote))
    for entry in iter_skills_files(container_base=container_base):
        files.append((entry["host_path"], entry["container_path"]))
    for entry in iter_cache_files(container_base=container_base):
        files.append((entry["host_path"], entry["container_path"]))
    return files


def quoted_rm_command(remote_paths: list[str]) -> str:
    """Build a shell ``rm -f`` command for a batch of remote paths."""
    return "rm -f " + " ".join(shlex.quote(p) for p in remote_paths)


def quoted_mkdir_command(dirs: list[str]) -> str:
    """Build a shell ``mkdir -p`` command for a batch of directories."""
    return "mkdir -p " + " ".join(shlex.quote(d) for d in dirs)


def unique_parent_dirs(files: list[tuple[str, str]]) -> list[str]:
    """Extract sorted unique parent directories from (host, remote) pairs."""
    return sorted({str(Path(remote).parent) for _, remote in files})


def _sha256_file(path: str) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _open_sync_back_destination(host_path: str, mapped_dir: str) -> tuple[int, bool]:
    """Open a mapped file without following untrusted descendant directories.

    Keep directory descriptors open while traversing so a remote-controlled
    nested path cannot escape through an existing or swapped parent symlink.
    """
    relative = os.path.relpath(host_path, mapped_dir)
    parts = Path(relative).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe sync-back destination: {host_path!r}")

    # A path-based fallback can race a replacement of even a mapped leaf.
    # Refuse all sync-back writes when handle-safe traversal is unavailable.
    if os.name != "posix" or not getattr(os, "O_NOFOLLOW", 0) or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("secure sync-back requires POSIX O_NOFOLLOW and O_DIRECTORY")

    dir_fd = None
    try:
        # The mapped directory itself is an explicit, trusted input. Resolve
        # its existing location once; only descendants come from remote paths.
        root = os.path.realpath(mapped_dir)
        os.makedirs(root, exist_ok=True)
        dir_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o700, dir_fd=dir_fd)
            except FileExistsError:
                pass
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd

        leaf = parts[-1]
        try:
            fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
            return fd, False
        except FileExistsError:
            fd = os.open(leaf, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
            return fd, True
    except OSError as exc:
        raise ValueError(f"unsafe sync-back destination: {host_path!r}") from exc
    finally:
        if dir_fd is not None:
            os.close(dir_fd)


_SYNC_BACK_MAX_RETRIES = 3
_SYNC_BACK_BACKOFF = (2, 4, 8)  # seconds between retries
_SYNC_BACK_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB — refuse to extract larger tars
_SYNC_BACK_MAX_MEMBERS = 100_000  # Bound tar metadata even if entries contain no data.
_SYNC_BACK_MAX_METADATA_BYTES = 64 * 1024 * 1024
_SYNC_BACK_MAX_METADATA_READ_BYTES = 1024 * 1024


class _BoundedTarReader:
    """Limit parser metadata reads before tarfile can allocate a PAX record."""

    def __init__(self, source):
        self.source = source
        self.metadata_bytes = 0
        self.validated = False

    def read(self, size=-1):
        if not self.validated:
            if (size < 0 or size > _SYNC_BACK_MAX_METADATA_READ_BYTES
                    or self.metadata_bytes + size > _SYNC_BACK_MAX_METADATA_BYTES):
                raise ValueError("sync-back archive metadata exceeds size cap")
        data = self.source.read(size)
        if not self.validated:
            self.metadata_bytes += len(data)
        return data

    def seek(self, offset, whence=0):
        return self.source.seek(offset, whence)

    def tell(self):
        return self.source.tell()


def _extract_sync_back_tar(tar: tarfile.TarFile, staging: str) -> dict[str, bool]:
    """Extract only bounded regular files and directories beneath staging.

    Python 3.11.0 lacks tarfile's ``filter='data'`` extraction argument.
    Validate the full archive before writing, independent of Python version.
    """
    # getmembers() eagerly loads unbounded metadata. Stop while iterating so
    # a compressed archive with many empty entries cannot exhaust memory.
    members: list[tarfile.TarInfo] = []
    total_bytes = 0
    seen: set[str] = set()
    for member in tar:
        if len(members) >= _SYNC_BACK_MAX_MEMBERS:
            raise ValueError("sync-back archive has too many members")
        members.append(member)
        name = member.name.rstrip("/") if member.isdir() else member.name
        parts = name.split("/")
        if (not name or name.startswith(("/", "\\")) or "\\" in name
                or ntpath.splitdrive(name)[0] or any(p in ("", ".", "..") for p in parts)
                or not (member.isdir() or member.isfile()) or name in seen):
            raise ValueError(f"unsafe sync-back archive member: {member.name!r}")
        seen.add(name)
        if member.isfile():
            if member.size < 0:
                raise ValueError("sync-back archive has negative file size")
            total_bytes += member.size
            if total_bytes > _SYNC_BACK_MAX_BYTES:
                raise ValueError("sync-back archive expands beyond size cap")
        # Python copies a global PAX dictionary onto EVERY member. A single
        # large header plus 100k empty members otherwise retains gigabytes.
        # Parsed name, size, and mode above remain on TarInfo after clearing.
        member.pax_headers = {}

    # Only the verified regular-file bytes remain. The parser has finished
    # reading metadata, so large but size-bounded payloads can now be copied.
    if isinstance(tar.fileobj, _BoundedTarReader):
        tar.fileobj.validated = True

    executable_files: dict[str, bool] = {}
    for member in members:
        target = os.path.join(staging, member.name)
        if member.isdir():
            os.makedirs(target, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read sync-back archive member: {member.name!r}")
            # Tar modes are untrusted. Stage private even when the host umask
            # would otherwise expose an API key to other local users.
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with source, os.fdopen(fd, "wb") as dest:
                shutil.copyfileobj(source, dest)
            executable_files[member.name] = bool(member.mode & 0o100)
    return executable_files


class FileSyncManager:
    """Tracks local file changes and syncs to a remote environment.

    Backends instantiate this with transport callbacks (upload, delete)
    and a file-source callable.  The manager handles mtime-based change
    detection, deletion tracking, rate limiting, and transactional state.

    Not used by bind-mount backends (Docker, Singularity) — those get
    live host FS views and don't need file sync.
    """

    def __init__(
        self,
        get_files_fn: GetFilesFn,
        upload_fn: UploadFn,
        delete_fn: DeleteFn,
        sync_interval: float = _SYNC_INTERVAL_SECONDS,
        bulk_upload_fn: BulkUploadFn | None = None,
        bulk_download_fn: BulkDownloadFn | None = None,
    ):
        self._get_files_fn = get_files_fn
        self._upload_fn = upload_fn
        self._bulk_upload_fn = bulk_upload_fn
        self._bulk_download_fn = bulk_download_fn
        self._delete_fn = delete_fn
        self._synced_files: dict[str, tuple[float, int]] = {}  # remote_path -> (mtime, size)
        self._pushed_hashes: dict[str, str] = {}  # remote_path -> sha256 hex digest
        self._last_sync_time: float = 0.0  # monotonic; 0 ensures first sync runs
        self._sync_interval = sync_interval

    def sync(self, *, force: bool = False) -> None:
        """Run a sync cycle: upload changed files, delete removed files.

        Rate-limited to once per ``sync_interval`` unless *force* is True
        or ``HERMES_FORCE_FILE_SYNC=1`` is set.

        Transactional: state only committed if ALL operations succeed.
        On failure, state rolls back so the next cycle retries everything.
        """
        if not force and not os.environ.get(_FORCE_SYNC_ENV):
            now = time.monotonic()
            if now - self._last_sync_time < self._sync_interval:
                return

        current_files = self._get_files_fn()
        current_remote_paths = {remote for _, remote in current_files}

        # --- Uploads: new or changed files ---
        to_upload: list[tuple[str, str]] = []
        new_files = dict(self._synced_files)
        for host_path, remote_path in current_files:
            file_key = _file_mtime_key(host_path)
            if file_key is None:
                continue
            if self._synced_files.get(remote_path) == file_key:
                continue
            to_upload.append((host_path, remote_path))
            new_files[remote_path] = file_key

        # --- Deletes: synced paths no longer in current set ---
        to_delete = [p for p in self._synced_files if p not in current_remote_paths]

        if not to_upload and not to_delete:
            self._last_sync_time = time.monotonic()
            return

        # Snapshot for rollback (only when there's work to do)
        prev_files = dict(self._synced_files)
        prev_hashes = dict(self._pushed_hashes)

        if to_upload:
            logger.debug("file_sync: uploading %d file(s)", len(to_upload))
        if to_delete:
            logger.debug("file_sync: deleting %d stale remote file(s)", len(to_delete))

        try:
            if to_upload and self._bulk_upload_fn is not None:
                self._bulk_upload_fn(to_upload)
                logger.debug("file_sync: bulk-uploaded %d file(s)", len(to_upload))
            else:
                for host_path, remote_path in to_upload:
                    self._upload_fn(host_path, remote_path)
                    logger.debug("file_sync: uploaded %s -> %s", host_path, remote_path)

            if to_delete:
                self._delete_fn(to_delete)
                logger.debug("file_sync: deleted %s", to_delete)

            # --- Commit (all succeeded) ---
            for host_path, remote_path in to_upload:
                self._pushed_hashes[remote_path] = _sha256_file(host_path)

            for p in to_delete:
                new_files.pop(p, None)
                self._pushed_hashes.pop(p, None)

            self._synced_files = new_files
            self._last_sync_time = time.monotonic()

        except Exception as exc:
            self._synced_files = prev_files
            self._pushed_hashes = prev_hashes
            self._last_sync_time = time.monotonic()
            logger.warning("file_sync: sync failed, rolled back state: %s", exc)

    # ------------------------------------------------------------------
    # Sync-back: pull remote changes to host on teardown
    # ------------------------------------------------------------------

    def sync_back(self, hermes_home: Path | None = None) -> None:
        """Pull remote changes back to the host filesystem.

        Downloads the remote ``.hermes/`` directory as a tar archive,
        unpacks it, and applies only files that differ from what was
        originally pushed (based on SHA-256 content hashes).

        Protected against SIGINT (defers the signal until complete) and
        serialized across concurrent gateway sandboxes via file lock.
        """
        if self._bulk_download_fn is None:
            return

        # Nothing was ever committed through this manager — the initial
        # push failed or never ran. Skip sync_back to avoid retry storms
        # against an uninitialized remote .hermes/ directory.
        if not self._pushed_hashes and not self._synced_files:
            logger.debug("sync_back: no prior push state — skipping")
            return

        lock_path = (hermes_home or get_hermes_home()) / ".sync.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        last_exc: Exception | None = None
        for attempt in range(_SYNC_BACK_MAX_RETRIES):
            try:
                self._sync_back_once(lock_path)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _SYNC_BACK_MAX_RETRIES - 1:
                    delay = _SYNC_BACK_BACKOFF[attempt]
                    logger.warning(
                        "sync_back: attempt %d failed (%s), retrying in %ds",
                        attempt + 1, exc, delay,
                    )
                    _sleep(delay)

        logger.warning("sync_back: all %d attempts failed: %s", _SYNC_BACK_MAX_RETRIES, last_exc)

    def _sync_back_once(self, lock_path: Path) -> None:
        """Single sync-back attempt with SIGINT protection and file lock."""
        # signal.signal() only works from the main thread. In gateway
        # contexts cleanup() may run from a worker thread — skip SIGINT
        # deferral there rather than crashing.
        on_main_thread = threading.current_thread() is threading.main_thread()

        deferred_sigint: list[object] = []
        original_handler = None
        if on_main_thread:
            original_handler = signal.getsignal(signal.SIGINT)

            def _defer_sigint(signum, frame):
                deferred_sigint.append((signum, frame))
                logger.debug("sync_back: SIGINT deferred until sync completes")

            signal.signal(signal.SIGINT, _defer_sigint)
        try:
            self._sync_back_locked(lock_path)
        finally:
            if on_main_thread and original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)
                if deferred_sigint:
                    os.kill(os.getpid(), signal.SIGINT)

    def _sync_back_locked(self, lock_path: Path) -> None:
        """Sync-back under file lock (serializes concurrent gateways)."""
        if fcntl is None:
            # Windows: no flock — run without serialization
            self._sync_back_impl()
            return
        lock_fd = open(lock_path, "w", encoding="utf-8")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._sync_back_impl()
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
            lock_fd.close()

    def _sync_back_impl(self) -> None:
        """Download, diff, and apply remote changes to host."""
        if self._bulk_download_fn is None:
            raise RuntimeError("_sync_back_impl called without bulk_download_fn")

        # Cache file mapping once to avoid O(n*m) from repeated iteration
        try:
            file_mapping = list(self._get_files_fn())
        except Exception:
            file_mapping = []

        with tempfile.NamedTemporaryFile(suffix=".tar") as tf:
            self._bulk_download_fn(Path(tf.name))

            # Defensive size cap: a misbehaving sandbox could produce an
            # arbitrarily large tar. Refuse to extract if it exceeds the cap.
            try:
                tar_size = os.path.getsize(tf.name)
            except OSError:
                tar_size = 0
            if tar_size > _SYNC_BACK_MAX_BYTES:
                logger.warning(
                    "sync_back: remote tar is %d bytes (cap %d) — skipping extraction",
                    tar_size, _SYNC_BACK_MAX_BYTES,
                )
                return

            with tempfile.TemporaryDirectory(prefix="hermes-sync-back-") as staging:
                # Every sync-back transport emits uncompressed `tar cf`.
                # r: rejects compressed archives before decompression can
                # inflate metadata beyond the downloaded-file size limit.
                with open(tf.name, "rb") as archive:
                    prefix = archive.read(6)
                    archive.seek(0)
                    if (prefix.startswith((b"\x1f\x8b", b"BZh"))
                            or prefix == b"\xfd7zXZ\x00"):
                        raise ValueError("compressed sync-back archives are not supported")
                    reader = _BoundedTarReader(archive)
                    with tarfile.open(fileobj=reader, mode="r:") as tar:
                        executable_files = _extract_sync_back_tar(tar, staging)

                applied = 0
                for dirpath, _dirnames, filenames in os.walk(staging):
                    for fname in filenames:
                        staged_file = os.path.join(dirpath, fname)
                        rel = os.path.relpath(staged_file, staging)
                        remote_path = "/" + rel

                        pushed_hash = self._pushed_hashes.get(remote_path)

                        # Skip hashing for files unchanged from push
                        if pushed_hash is not None:
                            remote_hash = _sha256_file(staged_file)
                            if remote_hash == pushed_hash:
                                continue
                        else:
                            remote_hash = None  # new remote file

                        # Resolve host path from cached mapping
                        host_path = self._resolve_host_path(remote_path, file_mapping)
                        mapped_dir = os.path.dirname(host_path) if host_path is not None else None
                        if host_path is None:
                            inferred = self._infer_host_target(remote_path, file_mapping)
                            if inferred is None:
                                logger.debug(
                                    "sync_back: skipping %s (no host mapping)",
                                    remote_path,
                                )
                                continue
                            host_path, mapped_dir = inferred

                        if os.path.exists(host_path) and pushed_hash is not None:
                            host_hash = _sha256_file(host_path)
                            if host_hash != pushed_hash:
                                logger.warning(
                                    "sync_back: conflict on %s — host modified "
                                    "since push, remote also changed. Applying "
                                    "remote version (last-write-wins).",
                                    remote_path,
                                )

                        # Create private BEFORE writing. copy2's copyfile
                        # creates mode 0666 until copystat runs, exposing
                        # credentials during the copy under a permissive umask.
                        fd, existed = _open_sync_back_destination(host_path, mapped_dir)
                        try:
                            if not stat.S_ISREG(os.fstat(fd).st_mode):
                                raise ValueError(f"unsafe sync-back destination: {host_path!r}")
                            if not existed and executable_files.get(rel, False):
                                os.fchmod(fd, 0o700)
                            os.ftruncate(fd, 0)
                            with open(staged_file, "rb") as source, os.fdopen(fd, "wb") as dest:
                                fd = -1  # fdopen now owns the descriptor.
                                shutil.copyfileobj(source, dest)
                        finally:
                            if fd >= 0:
                                os.close(fd)
                        applied += 1

                if applied:
                    logger.info("sync_back: applied %d changed file(s)", applied)
                else:
                    logger.debug("sync_back: no remote changes detected")

    def _resolve_host_path(self, remote_path: str,
                           file_mapping: list[tuple[str, str]] | None = None) -> str | None:
        """Find the host path for a known remote path from the file mapping."""
        mapping = file_mapping if file_mapping is not None else []
        for host, remote in mapping:
            if remote == remote_path:
                return host
        return None

    def _infer_host_path(self, remote_path: str,
                         file_mapping: list[tuple[str, str]] | None = None) -> str | None:
        """Infer a host path for a new remote file by matching path prefixes.

        Uses the existing file mapping to find a remote->host directory
        pair, then applies the same prefix substitution to the new file.
        For example, if the mapping has ``/root/.hermes/skills/a.md`` →
        ``~/.hermes/skills/a.md``, a new remote file at
        ``/root/.hermes/skills/b.md`` maps to ``~/.hermes/skills/b.md``.
        """
        target = self._infer_host_target(remote_path, file_mapping)
        return target[0] if target is not None else None

    def _infer_host_target(self, remote_path: str,
                           file_mapping: list[tuple[str, str]] | None = None) -> tuple[str, str] | None:
        """Return an inferred destination and the explicitly mapped parent."""
        mapping = file_mapping if file_mapping is not None else []
        for host, remote in mapping:
            remote_dir = str(Path(remote).parent)
            if remote_path.startswith(remote_dir + "/"):
                host_dir = str(Path(host).parent)
                suffix = remote_path[len(remote_dir):]
                return host_dir + suffix, host_dir
        return None
