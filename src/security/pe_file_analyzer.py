"""Portable Executable (PE) file header analyzer.

Pure-stdlib parser for PE/COFF files. Parses DOS header, NT headers (File +
Optional), section headers, and best-effort import directory. Computes
Shannon entropy per section to flag packer-like high-entropy regions.

References:
    Microsoft PE/COFF specification.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field


class InvalidPEError(ValueError):
    """Raised when input bytes are not a recognizable PE file."""


# IMAGE_FILE_MACHINE_* constants (subset)
_MACHINE_MAP = {
    0x0000: "unknown",
    0x014C: "x86",
    0x0200: "ia64",
    0x8664: "x64",
    0x01C0: "arm",
    0x01C4: "armnt",
    0xAA64: "arm64",
}

# IMAGE_FILE_DLL characteristic
_IMAGE_FILE_DLL = 0x2000

_PE_SIGNATURE = b"PE\x00\x00"
_DOS_MAGIC = b"MZ"


@dataclass
class PESection:
    name: str
    virtual_address: int
    virtual_size: int
    raw_size: int
    raw_offset: int
    entropy: float
    characteristics: int


@dataclass
class PEInfo:
    magic: str
    machine: str
    timestamp: int
    num_sections: int
    entry_point: int
    image_base: int
    sections: list[PESection] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    is_dll: bool = False
    suspicious: list[str] = field(default_factory=list)


def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    ent = 0.0
    for c in counts:
        if c:
            p = c / n
            ent -= p * math.log2(p)
    return ent


def _rva_to_offset(rva: int, sections: list[PESection]) -> int:
    for s in sections:
        if s.virtual_address <= rva < s.virtual_address + max(s.virtual_size, s.raw_size):
            return s.raw_offset + (rva - s.virtual_address)
    return -1


def _read_cstr(data: bytes, offset: int, limit: int = 512) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\x00", offset, min(len(data), offset + limit))
    if end == -1:
        end = min(len(data), offset + limit)
    try:
        return data[offset:end].decode("ascii", errors="replace")
    except Exception:
        return ""


def _parse_imports(
    data: bytes, import_rva: int, import_size: int, sections: list[PESection]
) -> list[str]:
    """Best-effort parse of IMAGE_DIRECTORY_ENTRY_IMPORT (import descriptors)."""
    if import_rva == 0 or import_size == 0:
        return []
    start = _rva_to_offset(import_rva, sections)
    if start < 0:
        return []
    dlls: list[str] = []
    # IMAGE_IMPORT_DESCRIPTOR is 20 bytes: 5 x uint32
    desc_size = 20
    offset = start
    max_entries = 512
    for _ in range(max_entries):
        if offset + desc_size > len(data):
            break
        chunk = data[offset : offset + desc_size]
        if len(chunk) < desc_size:
            break
        (orig_first_thunk, time_date, forwarder, name_rva, first_thunk) = struct.unpack(
            "<IIIII", chunk
        )
        # Null descriptor terminates the array
        if (
            orig_first_thunk == 0
            and time_date == 0
            and forwarder == 0
            and name_rva == 0
            and first_thunk == 0
        ):
            break
        if name_rva:
            name_off = _rva_to_offset(name_rva, sections)
            if name_off >= 0:
                name = _read_cstr(data, name_off, limit=256)
                if name:
                    dlls.append(name)
        offset += desc_size
    return dlls


def analyze_pe(data: bytes) -> PEInfo:
    """Analyze raw PE bytes and return :class:`PEInfo`.

    Raises ``InvalidPEError`` on malformed or non-PE input.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise InvalidPEError("input must be bytes-like")
    data = bytes(data)
    if len(data) < 64:
        raise InvalidPEError("input too short for DOS header")
    if data[:2] != _DOS_MAGIC:
        raise InvalidPEError("missing DOS 'MZ' magic")

    # e_lfanew at offset 0x3C (uint32)
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if e_lfanew <= 0 or e_lfanew + 24 > len(data):
        raise InvalidPEError("invalid e_lfanew offset")

    if data[e_lfanew : e_lfanew + 4] != _PE_SIGNATURE:
        raise InvalidPEError("missing 'PE\\0\\0' signature")

    # IMAGE_FILE_HEADER (20 bytes) immediately follows PE signature
    file_hdr_off = e_lfanew + 4
    if file_hdr_off + 20 > len(data):
        raise InvalidPEError("truncated file header")
    (
        machine,
        num_sections,
        timestamp,
        _sym_ptr,
        _num_syms,
        size_opt_hdr,
        characteristics,
    ) = struct.unpack_from("<HHIIIHH", data, file_hdr_off)

    opt_hdr_off = file_hdr_off + 20
    if size_opt_hdr == 0 or opt_hdr_off + size_opt_hdr > len(data):
        raise InvalidPEError("invalid/truncated optional header")

    opt_magic = struct.unpack_from("<H", data, opt_hdr_off)[0]
    suspicious: list[str] = []

    # Layout differs between PE32 (0x10B) and PE32+ (0x20B).
    # We pull: AddressOfEntryPoint, ImageBase, NumberOfRvaAndSizes, then data dirs.
    if opt_magic == 0x10B:
        # PE32
        # Entry point at +16 (uint32), ImageBase at +28 (uint32)
        entry_point = struct.unpack_from("<I", data, opt_hdr_off + 16)[0]
        image_base = struct.unpack_from("<I", data, opt_hdr_off + 28)[0]
        num_rva_off = opt_hdr_off + 92
        data_dirs_off = opt_hdr_off + 96
    elif opt_magic == 0x20B:
        # PE32+
        entry_point = struct.unpack_from("<I", data, opt_hdr_off + 16)[0]
        image_base = struct.unpack_from("<Q", data, opt_hdr_off + 24)[0]
        num_rva_off = opt_hdr_off + 108
        data_dirs_off = opt_hdr_off + 112
    else:
        # Still proceed for section parsing but record
        suspicious.append(f"unknown_optional_magic:0x{opt_magic:04x}")
        entry_point = 0
        image_base = 0
        num_rva_off = opt_hdr_off + 92
        data_dirs_off = opt_hdr_off + 96

    num_rva = 0
    import_rva = 0
    import_size = 0
    if num_rva_off + 4 <= len(data):
        num_rva = struct.unpack_from("<I", data, num_rva_off)[0]
    # Import directory is index 1 (Export=0, Import=1)
    if num_rva >= 2 and data_dirs_off + 16 <= len(data):
        import_rva, import_size = struct.unpack_from("<II", data, data_dirs_off + 8)

    # Section headers start at opt_hdr_off + size_opt_hdr
    sections_off = opt_hdr_off + size_opt_hdr
    sections: list[PESection] = []
    section_hdr_size = 40
    for i in range(num_sections):
        off = sections_off + i * section_hdr_size
        if off + section_hdr_size > len(data):
            suspicious.append("truncated_section_headers")
            break
        raw = data[off : off + section_hdr_size]
        name_bytes = raw[0:8]
        name = name_bytes.rstrip(b"\x00").decode("ascii", errors="replace")
        (
            virtual_size,
            virtual_address,
            raw_size,
            raw_offset,
            _reloc_ptr,
            _lineno_ptr,
            _num_reloc,
            _num_lineno,
            section_chars,
        ) = struct.unpack("<IIIIIIHHI", raw[8:])

        # Slice out section bytes for entropy
        entropy = 0.0
        if raw_size > 0 and 0 <= raw_offset < len(data):
            end = min(len(data), raw_offset + raw_size)
            section_bytes = data[raw_offset:end]
            entropy = _shannon_entropy(section_bytes)
        sections.append(
            PESection(
                name=name,
                virtual_address=virtual_address,
                virtual_size=virtual_size,
                raw_size=raw_size,
                raw_offset=raw_offset,
                entropy=entropy,
                characteristics=section_chars,
            )
        )
        if entropy > 7.0:
            suspicious.append(f"high_entropy_section:{name or f'#{i}'}:{entropy:.2f}")

    # Imports (best-effort)
    imports: list[str] = []
    try:
        imports = _parse_imports(data, import_rva, import_size, sections)
    except Exception as exc:  # pragma: no cover - defensive
        suspicious.append(f"import_parse_failed:{type(exc).__name__}")
        imports = []

    machine_name = _MACHINE_MAP.get(machine, "unknown")
    is_dll = bool(characteristics & _IMAGE_FILE_DLL)

    return PEInfo(
        magic="PE",
        machine=machine_name,
        timestamp=timestamp,
        num_sections=num_sections,
        entry_point=entry_point,
        image_base=image_base,
        sections=sections,
        imports=imports,
        is_dll=is_dll,
        suspicious=suspicious,
    )


def analyze_pe_file(path: str) -> PEInfo:
    """Read the file at ``path`` and return :class:`PEInfo`."""
    with open(path, "rb") as fh:
        data = fh.read()
    return analyze_pe(data)


__all__ = [
    "InvalidPEError",
    "PESection",
    "PEInfo",
    "analyze_pe",
    "analyze_pe_file",
]
