"""Safe archive extraction (zip-slip and tar-slip prevention). AUR-SEC-2026-0014. STRIDE: Tampering, EoP. CWE-22."""  # noqa: E501

import logging
import tarfile
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)


class ArchiveError(Exception):
    """Raised on any unsafe or malformed archive condition."""


class SafeZipExtractor:
    """Extracts ZIP archives member-by-member with path-traversal and zip-bomb guards.

    Never calls ``ZipFile.extractall()``; every member is validated before
    any bytes are written to disk.
    """

    def __init__(
        self,
        dest_dir: str | Path,
        max_file_size_mb: float = 100.0,
        max_total_mb: float = 500.0,
        max_files: int = 10_000,
    ) -> None:
        self.dest_dir = Path(dest_dir).resolve()
        self.max_file_size_bytes = int(max_file_size_mb * 1024**2)
        self.max_total_bytes = int(max_total_mb * 1024**2)
        self.max_files = max_files

    def extract(self, archive_path: str | Path) -> list[Path]:
        """Extract *archive_path* to ``dest_dir`` with full safety validation.

        Returns a list of extracted :class:`~pathlib.Path` objects.
        Raises :class:`ArchiveError` on the first unsafe or oversized member.
        """
        archive_path = Path(archive_path)
        try:
            zf = zipfile.ZipFile(archive_path, "r")  # noqa: SIM115
        except (zipfile.BadZipFile, OSError) as exc:
            raise ArchiveError(f"Cannot open ZIP archive {archive_path}: {exc}") from exc

        with zf:
            members = zf.infolist()

            if len(members) > self.max_files:
                raise ArchiveError(
                    f"Archive contains {len(members)} members, limit is {self.max_files}"
                )

            extracted: list[Path] = []
            total_bytes = 0

            for info in members:
                name = info.filename

                # Reject absolute paths immediately (zip-slip vector)
                if name.startswith("/") or name.startswith("\\"):
                    log.warning("Skipping ZIP member with absolute path: %r", name)
                    continue

                # Reject any path containing '..' components (zip-slip vector)
                parts = Path(name).parts
                if ".." in parts:
                    raise ArchiveError(
                        f"Zip-slip detected: member {name!r} contains '..' path component"
                    )

                # Resolve the final destination and assert containment (zip-slip guard)
                target = (self.dest_dir / name).resolve()
                if not str(target).startswith(str(self.dest_dir) + "/") and target != self.dest_dir:
                    raise ArchiveError(
                        f"Zip-slip detected: member {name!r} would extract to {target}, "
                        f"outside of {self.dest_dir}"
                    )

                # Per-file size guard (uses file_size = uncompressed size)
                if info.file_size > self.max_file_size_bytes:
                    raise ArchiveError(
                        f"Member {name!r} uncompressed size {info.file_size} bytes "
                        f"exceeds per-file limit of {self.max_file_size_bytes} bytes"
                    )

                # Running total guard
                total_bytes += info.file_size
                if total_bytes > self.max_total_bytes:
                    raise ArchiveError(
                        f"Total extraction size exceeded {self.max_total_bytes} bytes "
                        f"after member {name!r}"
                    )

                # Directory entry — just create it
                if name.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                # Extract single member safely
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())

                extracted.append(target)

        return extracted


class SafeTarExtractor:
    """Extracts TAR archives member-by-member with path-traversal and tar-slip guards.

    Supports ``.tar``, ``.tar.gz``, ``.tar.bz2``, and ``.tar.xz``.
    Never calls ``TarFile.extractall()``; every member is validated before
    any bytes are written to disk.
    """

    def __init__(
        self,
        dest_dir: str | Path,
        max_file_size_mb: float = 100.0,
        max_total_mb: float = 500.0,
        max_files: int = 10_000,
    ) -> None:
        self.dest_dir = Path(dest_dir).resolve()
        self.max_file_size_bytes = int(max_file_size_mb * 1024**2)
        self.max_total_bytes = int(max_total_mb * 1024**2)
        self.max_files = max_files

    def _assert_within_dest(self, name: str, target: Path) -> None:
        """Raise ArchiveError if *target* is not under ``dest_dir``."""
        dest_str = str(self.dest_dir)
        target_str = str(target)
        if not (target_str.startswith(dest_str + "/") or target_str == dest_str):
            raise ArchiveError(
                f"Tar-slip detected: member {name!r} would extract to {target}, "
                f"outside of {self.dest_dir}"
            )

    def extract(self, archive_path: str | Path) -> list[Path]:
        """Extract *archive_path* to ``dest_dir`` with full safety validation.

        Returns a list of extracted :class:`~pathlib.Path` objects.
        Raises :class:`ArchiveError` on the first unsafe member or size violation.
        """
        archive_path = Path(archive_path)
        try:
            tf = tarfile.open(archive_path, "r:*")  # noqa: SIM115
        except (tarfile.TarError, OSError) as exc:
            raise ArchiveError(f"Cannot open TAR archive {archive_path}: {exc}") from exc

        with tf:
            members = tf.getmembers()

            if len(members) > self.max_files:
                raise ArchiveError(
                    f"Archive contains {len(members)} members, limit is {self.max_files}"
                )

            extracted: list[Path] = []
            total_bytes = 0

            for member in members:
                name = member.name

                # Reject members with .. in name (tar-slip via path components)
                parts = Path(name).parts
                if ".." in parts:
                    raise ArchiveError(f"Tar-slip detected: member name contains '..': {name!r}")

                # Reject device files (block/char devices, fifos)
                if member.isdev() or member.isfifo():
                    raise ArchiveError(f"Rejecting device/FIFO tar member: {name!r}")

                # Reject hardlinks — can overwrite arbitrary files
                if member.islnk():
                    raise ArchiveError(
                        f"Rejecting hardlink tar member: {name!r} -> {member.linkname!r}"
                    )

                # Resolve destination path and assert containment (tar-slip guard)
                target = (self.dest_dir / name).resolve()
                self._assert_within_dest(name, target)

                # Validate symlink destination
                if member.issym():
                    # Resolve symlink relative to its containing directory
                    link_target = (target.parent / member.linkname).resolve()
                    dest_str = str(self.dest_dir)
                    link_str = str(link_target)
                    if not (link_str.startswith(dest_str + "/") or link_str == dest_str):
                        raise ArchiveError(
                            f"Symlink {name!r} -> {member.linkname!r} points outside "
                            f"dest_dir {self.dest_dir}"
                        )
                    # Create the symlink
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(member.linkname)
                    extracted.append(target)
                    continue

                # Directory entry
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                # Regular file — enforce size limits before extracting
                if member.size > self.max_file_size_bytes:
                    raise ArchiveError(
                        f"Member {name!r} size {member.size} bytes "
                        f"exceeds per-file limit of {self.max_file_size_bytes} bytes"
                    )

                total_bytes += member.size
                if total_bytes > self.max_total_bytes:
                    raise ArchiveError(
                        f"Total extraction size exceeded {self.max_total_bytes} bytes "
                        f"after member {name!r}"
                    )

                # Extract single member safely
                target.parent.mkdir(parents=True, exist_ok=True)
                file_obj = tf.extractfile(member)
                if file_obj is None:
                    log.warning("Skipping non-regular tar member: %r", name)
                    continue
                with file_obj, open(target, "wb") as dst:
                    dst.write(file_obj.read())

                extracted.append(target)

        return extracted


SAFE_EXTRACTOR_REGISTRY: dict[str, type] = {
    "zip": SafeZipExtractor,
    "tar": SafeTarExtractor,
}


def safe_extract(
    archive_path: str | Path,
    dest_dir: str | Path,
    **kwargs,
) -> list[Path]:
    """Auto-detect archive format and extract safely.

    Dispatches to :class:`SafeZipExtractor` for ``.zip`` files and
    :class:`SafeTarExtractor` for ``.tar``, ``.tar.gz``, ``.tar.bz2``,
    and ``.tar.xz`` files.

    Raises :class:`ArchiveError` for unknown formats or any safety violation.
    """
    path = Path(archive_path)
    name_lower = path.name.lower()

    if name_lower.endswith(".zip"):
        zip_extractor = SafeZipExtractor(dest_dir, **kwargs)
        return zip_extractor.extract(path)

    tar_suffixes = (".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz")
    if any(name_lower.endswith(s) for s in tar_suffixes):
        tar_extractor = SafeTarExtractor(dest_dir, **kwargs)
        return tar_extractor.extract(path)

    raise ArchiveError(
        f"Unknown or unsupported archive format: {path.name!r}. "
        f"Supported: .zip, .tar, .tar.gz, .tar.bz2, .tar.xz"
    )
