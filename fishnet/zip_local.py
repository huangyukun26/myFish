from __future__ import annotations

import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional


LOCAL_FILE_HEADER = b"PK\x03\x04"


@dataclass(frozen=True)
class LocalZipEntry:
    name: str
    method: int
    compressed_size: int
    uncompressed_size: int
    data_offset: int
    header_offset: int
    crc32: int


def iter_local_entries(zip_path: os.PathLike[str] | str, max_bytes: Optional[int] = None) -> Iterator[LocalZipEntry]:
    """Iterate complete local ZIP entries without requiring the central directory.

    This is useful while a large ZIP is still incomplete. It supports entries whose
    local header already contains compressed sizes, which is true for the current
    FishNet images archive.
    """
    path = Path(zip_path)
    file_size = path.stat().st_size
    readable_size = min(file_size, max_bytes) if max_bytes is not None else file_size

    with path.open("rb") as fp:
        offset = 0
        while offset + 30 <= readable_size:
            fp.seek(offset)
            header = fp.read(30)
            if len(header) < 30 or header[:4] != LOCAL_FILE_HEADER:
                break

            (
                _signature,
                _version,
                flags,
                method,
                _mtime,
                _mdate,
                crc32,
                compressed_size,
                uncompressed_size,
                name_len,
                extra_len,
            ) = struct.unpack("<IHHHHHIIIHH", header)

            if flags & 0x08:
                # Data descriptor mode stores sizes after data; avoid guessing.
                break

            raw_name = fp.read(name_len)
            fp.seek(extra_len, os.SEEK_CUR)
            try:
                name = raw_name.decode("utf-8")
            except UnicodeDecodeError:
                name = raw_name.decode("cp437", errors="replace")

            data_offset = offset + 30 + name_len + extra_len
            next_offset = data_offset + compressed_size
            if next_offset > readable_size:
                break

            yield LocalZipEntry(
                name=name,
                method=method,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                data_offset=data_offset,
                header_offset=offset,
                crc32=crc32,
            )
            offset = next_offset


def read_entry_data(fp: BinaryIO, entry: LocalZipEntry) -> bytes:
    fp.seek(entry.data_offset)
    data = fp.read(entry.compressed_size)
    if entry.method == 0:
        return data
    if entry.method == 8:
        return zlib.decompress(data, -15)
    raise ValueError(f"Unsupported ZIP compression method {entry.method} for {entry.name}")

