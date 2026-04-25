"""Tests for src.security.safe_archive — zip-slip, tar-slip, and zip-bomb prevention.

AUR-SEC-2026-0014. STRIDE: Tampering, EoP. CWE-22.
"""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from src.security.safe_archive import (
    ArchiveError,
    SAFE_EXTRACTOR_REGISTRY,
    SafeTarExtractor,
    SafeZipExtractor,
    safe_extract,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zip(path: Path, members: list[tuple[str, bytes]]) -> Path:
    """Write a ZIP file at *path* with the given (name, data) members."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return path


def _make_tar(path: Path, members: list[tuple[str, bytes]]) -> Path:
    """Write an uncompressed TAR at *path* with the given (name, data) members."""
    with tarfile.open(path, "w") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


# ---------------------------------------------------------------------------
# Normal extraction
# ---------------------------------------------------------------------------


class TestNormalZipExtraction:
    def test_extracts_files_within_dest(self, tmp_path):
        archive = _make_zip(
            tmp_path / "archive.zip",
            [("hello.txt", b"hello"), ("subdir/world.txt", b"world")],
        )
        dest = tmp_path / "out"
        dest.mkdir()

        extractor = SafeZipExtractor(dest)
        result = extractor.extract(archive)

        paths = {p.relative_to(dest).as_posix() for p in result}
        assert paths == {"hello.txt", "subdir/world.txt"}
        assert (dest / "hello.txt").read_bytes() == b"hello"
        assert (dest / "subdir" / "world.txt").read_bytes() == b"world"
        # All returned paths are within dest_dir
        for p in result:
            assert str(p).startswith(str(dest))

    def test_returns_list_of_paths(self, tmp_path):
        archive = _make_zip(tmp_path / "a.zip", [("file.bin", b"\x00" * 10)])
        dest = tmp_path / "out"
        dest.mkdir()
        result = SafeZipExtractor(dest).extract(archive)
        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)


class TestNormalTarExtraction:
    def test_extracts_files_within_dest(self, tmp_path):
        archive = _make_tar(
            tmp_path / "archive.tar",
            [("readme.txt", b"readme"), ("data/info.bin", b"\x01\x02")],
        )
        dest = tmp_path / "out"
        dest.mkdir()

        extractor = SafeTarExtractor(dest)
        result = extractor.extract(archive)

        paths = {p.relative_to(dest).as_posix() for p in result}
        assert paths == {"readme.txt", "data/info.bin"}
        assert (dest / "readme.txt").read_bytes() == b"readme"

    def test_returns_list_of_paths(self, tmp_path):
        archive = _make_tar(tmp_path / "a.tar", [("x.txt", b"x")])
        dest = tmp_path / "out"
        dest.mkdir()
        result = SafeTarExtractor(dest).extract(archive)
        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)


# ---------------------------------------------------------------------------
# Zip-slip guard
# ---------------------------------------------------------------------------


class TestZipSlip:
    def test_path_traversal_member_raises(self, tmp_path):
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            # Manually insert a member with a traversal path
            zf.writestr("../../../etc/passwd", "root:x:0:0:root:/root:/bin/bash")

        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ArchiveError, match="[Zz]ip.slip|outside"):
            SafeZipExtractor(dest).extract(archive)

    def test_dotdot_component_skipped_or_raises(self, tmp_path):
        """Members with '..' in path must not escape dest_dir."""
        archive = tmp_path / "evil2.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("safe/../../../sneaky.txt", "pwned")

        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ArchiveError):
            SafeZipExtractor(dest).extract(archive)


# ---------------------------------------------------------------------------
# Tar-slip guard
# ---------------------------------------------------------------------------


class TestTarSlip:
    def test_path_traversal_member_raises(self, tmp_path):
        archive = tmp_path / "evil.tar"
        with tarfile.open(archive, "w") as tf:
            info = tarfile.TarInfo(name="../../outside")
            info.size = 6
            tf.addfile(info, io.BytesIO(b"danger"))

        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ArchiveError, match="[Tt]ar.slip|outside|'\\.\\.'"):
            SafeTarExtractor(dest).extract(archive)


# ---------------------------------------------------------------------------
# Zip-bomb per-file guard
# ---------------------------------------------------------------------------


class TestZipBomb:
    def test_large_declared_file_size_raises(self, tmp_path):
        """Member with declared uncompressed size > max_file_size_mb must raise.

        We create a real ZIP, then monkey-patch ZipInfo.file_size on the infolist
        to simulate a declared-200MB entry without actually writing 200 MB to disk.
        This mirrors what a real zip-bomb does: compress_size is tiny but file_size
        (the uncompressed size stored in the central directory) is enormous.
        """
        from unittest.mock import patch

        archive = tmp_path / "bomb.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Actual content is tiny; we'll lie about its size via patch
            zf.writestr("bigfile.bin", b"A" * 1024)

        dest = tmp_path / "out"
        dest.mkdir()

        # 200 MB declared — well above the 100 MB limit
        declared_size = 200 * 1024 * 1024

        original_infolist = zipfile.ZipFile.infolist

        def patched_infolist(self):
            infos = original_infolist(self)
            for info in infos:
                if info.filename == "bigfile.bin":
                    info.file_size = declared_size
            return infos

        with patch.object(zipfile.ZipFile, "infolist", patched_infolist):
            with pytest.raises(ArchiveError, match="[Ss]ize|limit|exceed"):
                SafeZipExtractor(dest, max_file_size_mb=100.0).extract(archive)


# ---------------------------------------------------------------------------
# Too many files guard
# ---------------------------------------------------------------------------


class TestTooManyFiles:
    def test_exceeding_max_files_raises(self, tmp_path):
        max_files = 5
        archive = tmp_path / "many.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for i in range(max_files + 1):
                zf.writestr(f"file_{i:04d}.txt", b"x")

        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ArchiveError, match="[Mm]ember|[Ff]ile|limit|exceed"):
            SafeZipExtractor(dest, max_files=max_files).extract(archive)

    def test_tar_exceeding_max_files_raises(self, tmp_path):
        max_files = 3
        archive = tmp_path / "many.tar"
        with tarfile.open(archive, "w") as tf:
            for i in range(max_files + 1):
                data = b"y"
                info = tarfile.TarInfo(name=f"f{i}.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ArchiveError, match="[Mm]ember|[Ff]ile|limit|exceed"):
            SafeTarExtractor(dest, max_files=max_files).extract(archive)


# ---------------------------------------------------------------------------
# Symlink outside dest_dir
# ---------------------------------------------------------------------------


class TestSymlinkOutside:
    def test_symlink_to_outside_raises(self, tmp_path):
        archive = tmp_path / "symlink.tar"
        with tarfile.open(archive, "w") as tf:
            info = tarfile.TarInfo(name="evil_link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            info.size = 0
            tf.addfile(info)

        dest = tmp_path / "out"
        dest.mkdir()

        with pytest.raises(ArchiveError, match="[Ss]ymlink|outside|[Ss]lip"):
            SafeTarExtractor(dest).extract(archive)

    def test_symlink_within_dest_allowed(self, tmp_path):
        """A symlink pointing to a file within dest_dir should be accepted."""
        archive = tmp_path / "ok_symlink.tar"
        target_name = "real_file.txt"
        link_name = "link_to_real.txt"
        with tarfile.open(archive, "w") as tf:
            # First add the real file
            data = b"contents"
            info = tarfile.TarInfo(name=target_name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            # Then add symlink pointing to it (relative path, within dest)
            sym_info = tarfile.TarInfo(name=link_name)
            sym_info.type = tarfile.SYMTYPE
            sym_info.linkname = target_name
            sym_info.size = 0
            tf.addfile(sym_info)

        dest = tmp_path / "out"
        dest.mkdir()

        result = SafeTarExtractor(dest).extract(archive)
        result_names = {p.name for p in result}
        assert target_name in result_names
        assert link_name in result_names


# ---------------------------------------------------------------------------
# safe_extract() dispatch
# ---------------------------------------------------------------------------


class TestSafeExtractDispatch:
    def test_detects_zip_by_extension(self, tmp_path):
        archive = _make_zip(tmp_path / "data.zip", [("a.txt", b"hello")])
        dest = tmp_path / "out"
        dest.mkdir()
        result = safe_extract(archive, dest)
        assert any(p.name == "a.txt" for p in result)

    def test_detects_tar_by_extension(self, tmp_path):
        archive = _make_tar(tmp_path / "data.tar", [("b.txt", b"world")])
        dest = tmp_path / "out"
        dest.mkdir()
        result = safe_extract(archive, dest)
        assert any(p.name == "b.txt" for p in result)

    def test_detects_tar_gz_by_extension(self, tmp_path):
        archive = tmp_path / "data.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            data = b"compressed"
            info = tarfile.TarInfo(name="c.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        dest = tmp_path / "out"
        dest.mkdir()
        result = safe_extract(archive, dest)
        assert any(p.name == "c.txt" for p in result)

    def test_unknown_extension_raises(self, tmp_path):
        fake = tmp_path / "data.rar"
        fake.write_bytes(b"not an archive")
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(ArchiveError, match="[Uu]nsupported|[Uu]nknown"):
            safe_extract(fake, dest)


# ---------------------------------------------------------------------------
# Malformed archive
# ---------------------------------------------------------------------------


class TestMalformedArchive:
    def test_random_bytes_as_zip_raises_archive_error(self, tmp_path):
        bad = tmp_path / "junk.zip"
        bad.write_bytes(b"\xff\xfe\xfd\xfc" * 256)
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(ArchiveError):
            SafeZipExtractor(dest).extract(bad)

    def test_random_bytes_as_tar_raises_archive_error(self, tmp_path):
        bad = tmp_path / "junk.tar"
        bad.write_bytes(b"\x00" * 10)
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(ArchiveError):
            SafeTarExtractor(dest).extract(bad)

    def test_does_not_crash_with_empty_file(self, tmp_path):
        bad = tmp_path / "empty.zip"
        bad.write_bytes(b"")
        dest = tmp_path / "out"
        dest.mkdir()
        with pytest.raises(ArchiveError):
            SafeZipExtractor(dest).extract(bad)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_contains_zip_and_tar(self):
        assert "zip" in SAFE_EXTRACTOR_REGISTRY
        assert "tar" in SAFE_EXTRACTOR_REGISTRY
        assert SAFE_EXTRACTOR_REGISTRY["zip"] is SafeZipExtractor
        assert SAFE_EXTRACTOR_REGISTRY["tar"] is SafeTarExtractor
