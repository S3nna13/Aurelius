"""Tests for ``src.security.pe_file_analyzer``.

Synthetic PE bytes are hand-crafted per Microsoft's PE/COFF spec so tests
run on any platform without real executables.
"""

from __future__ import annotations

import os
import struct

import pytest

from src.security.pe_file_analyzer import (
    InvalidPEError,
    PEInfo,
    PESection,
    analyze_pe,
    analyze_pe_file,
)


# ---------------------------------------------------------------------------
# Synthetic PE builder
# ---------------------------------------------------------------------------

_DOS_STUB_SIZE = 0x40  # DOS header up to e_lfanew
_FILE_HDR_SIZE = 20
_OPT_HDR_PE32_SIZE = 224  # standard PE32 optional header (96 + 16*8)
_SECTION_HDR_SIZE = 40
_SECTION_ALIGN_RAW = 0x200


def build_minimal_pe(
    *,
    machine: int = 0x014C,  # x86
    timestamp: int = 0x5F5E_1000,
    entry_point: int = 0x1000,
    image_base: int = 0x0040_0000,
    is_dll: bool = False,
    section_name: bytes = b".text",
    section_body: bytes = b"\x00" * 32,
    extra_chars: int = 0,
) -> bytes:
    """Construct a minimal but structurally valid PE32 file with one section."""
    if len(section_name) > 8:
        raise ValueError("section name max 8 bytes")

    # Layout:
    #   [0x00 .. 0x3F]  DOS header (e_lfanew at 0x3C points to PE sig)
    #   [e_lfanew]      "PE\0\0"
    #   [+4]            IMAGE_FILE_HEADER
    #   [+24]           IMAGE_OPTIONAL_HEADER32 (224 bytes)
    #   [+248]          section headers
    #   [raw_offset]    section data (aligned up to 0x200)

    e_lfanew = _DOS_STUB_SIZE  # 64
    pe_sig_off = e_lfanew
    file_hdr_off = pe_sig_off + 4
    opt_hdr_off = file_hdr_off + _FILE_HDR_SIZE
    sections_off = opt_hdr_off + _OPT_HDR_PE32_SIZE
    num_sections = 1
    headers_end = sections_off + num_sections * _SECTION_HDR_SIZE

    # Align raw_offset up to 0x200
    raw_offset = ((headers_end + _SECTION_ALIGN_RAW - 1) // _SECTION_ALIGN_RAW) * _SECTION_ALIGN_RAW

    raw_size = len(section_body)

    # DOS header: MZ + padding up to 0x3C, then e_lfanew
    dos = bytearray(b"\x00" * _DOS_STUB_SIZE)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, e_lfanew)

    # File header
    characteristics = 0x0002  # EXECUTABLE_IMAGE
    if is_dll:
        characteristics |= 0x2000
    characteristics |= extra_chars
    file_hdr = struct.pack(
        "<HHIIIHH",
        machine,
        num_sections,
        timestamp,
        0,  # PointerToSymbolTable
        0,  # NumberOfSymbols
        _OPT_HDR_PE32_SIZE,
        characteristics,
    )

    # Optional header PE32 (magic 0x10B), 224 bytes
    opt = bytearray(_OPT_HDR_PE32_SIZE)
    struct.pack_into("<H", opt, 0, 0x010B)  # Magic PE32
    opt[2] = 1  # MajorLinkerVersion
    opt[3] = 0  # MinorLinkerVersion
    struct.pack_into("<I", opt, 4, raw_size)  # SizeOfCode
    struct.pack_into("<I", opt, 8, 0)  # SizeOfInitializedData
    struct.pack_into("<I", opt, 12, 0)  # SizeOfUninitializedData
    struct.pack_into("<I", opt, 16, entry_point)  # AddressOfEntryPoint
    struct.pack_into("<I", opt, 20, 0x1000)  # BaseOfCode
    struct.pack_into("<I", opt, 24, 0x2000)  # BaseOfData (PE32 only)
    struct.pack_into("<I", opt, 28, image_base)  # ImageBase
    struct.pack_into("<I", opt, 32, 0x1000)  # SectionAlignment
    struct.pack_into("<I", opt, 36, 0x200)  # FileAlignment
    struct.pack_into("<H", opt, 40, 6)  # MajorOSVersion
    struct.pack_into("<H", opt, 42, 0)
    struct.pack_into("<H", opt, 44, 0)
    struct.pack_into("<H", opt, 46, 0)
    struct.pack_into("<H", opt, 48, 6)  # MajorSubsystemVersion
    struct.pack_into("<H", opt, 50, 0)
    struct.pack_into("<I", opt, 52, 0)  # Win32VersionValue
    struct.pack_into("<I", opt, 56, 0x2000)  # SizeOfImage
    struct.pack_into("<I", opt, 60, raw_offset)  # SizeOfHeaders
    struct.pack_into("<I", opt, 64, 0)  # CheckSum
    struct.pack_into("<H", opt, 68, 3)  # Subsystem (console)
    struct.pack_into("<H", opt, 70, 0)  # DllCharacteristics
    struct.pack_into("<I", opt, 72, 0x100000)  # SizeOfStackReserve
    struct.pack_into("<I", opt, 76, 0x1000)
    struct.pack_into("<I", opt, 80, 0x100000)
    struct.pack_into("<I", opt, 84, 0x1000)
    struct.pack_into("<I", opt, 88, 0)  # LoaderFlags
    struct.pack_into("<I", opt, 92, 16)  # NumberOfRvaAndSizes
    # Data directories (16 * 8 = 128 bytes starting at offset 96) left zero.

    # Section header
    sec = bytearray(_SECTION_HDR_SIZE)
    name_padded = section_name.ljust(8, b"\x00")
    sec[0:8] = name_padded
    struct.pack_into("<I", sec, 8, raw_size)  # VirtualSize
    struct.pack_into("<I", sec, 12, 0x1000)  # VirtualAddress
    struct.pack_into("<I", sec, 16, raw_size)  # SizeOfRawData
    struct.pack_into("<I", sec, 20, raw_offset)  # PointerToRawData
    struct.pack_into("<I", sec, 24, 0)
    struct.pack_into("<I", sec, 28, 0)
    struct.pack_into("<H", sec, 32, 0)
    struct.pack_into("<H", sec, 34, 0)
    struct.pack_into("<I", sec, 36, 0x60000020)  # Characteristics: CODE|EXEC|READ

    # Assemble
    buf = bytearray(raw_offset + raw_size)
    buf[0:_DOS_STUB_SIZE] = dos
    buf[pe_sig_off : pe_sig_off + 4] = b"PE\x00\x00"
    buf[file_hdr_off : file_hdr_off + _FILE_HDR_SIZE] = file_hdr
    buf[opt_hdr_off : opt_hdr_off + _OPT_HDR_PE32_SIZE] = opt
    buf[sections_off : sections_off + _SECTION_HDR_SIZE] = sec
    buf[raw_offset : raw_offset + raw_size] = section_body
    return bytes(buf)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_minimal_pe_parses_to_peinfo() -> None:
    info = analyze_pe(build_minimal_pe())
    assert isinstance(info, PEInfo)
    assert len(info.sections) == 1
    assert isinstance(info.sections[0], PESection)


def test_magic_field_is_pe() -> None:
    info = analyze_pe(build_minimal_pe())
    assert info.magic == "PE"


@pytest.mark.parametrize(
    ("machine_code", "expected"),
    [
        (0x014C, "x86"),
        (0x8664, "x64"),
        (0x01C0, "arm"),
        (0xAA64, "arm64"),
        (0xDEAD, "unknown"),
    ],
)
def test_machine_field_resolves(machine_code: int, expected: str) -> None:
    info = analyze_pe(build_minimal_pe(machine=machine_code))
    assert info.machine == expected


def test_num_sections_correct() -> None:
    info = analyze_pe(build_minimal_pe())
    assert info.num_sections == 1


def test_section_entropy_in_range() -> None:
    info = analyze_pe(build_minimal_pe(section_body=b"hello world!" * 4))
    ent = info.sections[0].entropy
    assert 0.0 <= ent <= 8.0


def test_high_entropy_section_flagged_suspicious() -> None:
    # Pseudo-random section bytes should push entropy > 7.0
    body = bytes((i * 167 + 13) & 0xFF for i in range(2048))
    info = analyze_pe(build_minimal_pe(section_body=body))
    assert info.sections[0].entropy > 7.0
    assert any("high_entropy_section" in s for s in info.suspicious)


def test_low_entropy_section_not_flagged() -> None:
    info = analyze_pe(build_minimal_pe(section_body=b"\x00" * 1024))
    assert info.sections[0].entropy < 1.0
    assert not any("high_entropy_section" in s for s in info.suspicious)


def test_invalid_dos_magic_raises() -> None:
    data = bytearray(build_minimal_pe())
    data[0:2] = b"ZZ"
    with pytest.raises(InvalidPEError):
        analyze_pe(bytes(data))


def test_too_short_input_raises() -> None:
    with pytest.raises(InvalidPEError):
        analyze_pe(b"MZ")


def test_missing_pe_signature_raises() -> None:
    data = bytearray(build_minimal_pe())
    # Corrupt PE signature
    e_lfanew = struct.unpack_from("<I", bytes(data), 0x3C)[0]
    data[e_lfanew : e_lfanew + 4] = b"XX\x00\x00"
    with pytest.raises(InvalidPEError):
        analyze_pe(bytes(data))


def test_dll_flag_detected() -> None:
    info = analyze_pe(build_minimal_pe(is_dll=True))
    assert info.is_dll is True


def test_not_dll_flag() -> None:
    info = analyze_pe(build_minimal_pe(is_dll=False))
    assert info.is_dll is False


def test_timestamp_extracted() -> None:
    ts = 0x1234_5678
    info = analyze_pe(build_minimal_pe(timestamp=ts))
    assert info.timestamp == ts


def test_image_base_extracted() -> None:
    info = analyze_pe(build_minimal_pe(image_base=0x1000_0000))
    assert info.image_base == 0x1000_0000


def test_entry_point_extracted() -> None:
    info = analyze_pe(build_minimal_pe(entry_point=0x2468))
    assert info.entry_point == 0x2468


def test_determinism() -> None:
    data = build_minimal_pe()
    a = analyze_pe(data)
    b = analyze_pe(data)
    assert a == b


def test_section_names_stripped_of_null_padding() -> None:
    info = analyze_pe(build_minimal_pe(section_name=b".text"))
    # Section name should not contain embedded null bytes
    assert info.sections[0].name == ".text"
    assert "\x00" not in info.sections[0].name


def test_imports_empty_when_no_directory() -> None:
    info = analyze_pe(build_minimal_pe())
    assert info.imports == []


def test_analyze_pe_file_roundtrip(tmp_path) -> None:
    path = tmp_path / "synthetic.exe"
    path.write_bytes(build_minimal_pe())
    info = analyze_pe_file(str(path))
    assert info.magic == "PE"
    assert info.num_sections == 1


def test_rejects_non_bytes() -> None:
    with pytest.raises(InvalidPEError):
        analyze_pe("not bytes")  # type: ignore[arg-type]
