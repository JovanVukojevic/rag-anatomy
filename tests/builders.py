import hashlib
import io
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid7

import docx
from docx.document import Document as DocxDocument

from rag_anatomy.domain import Chunk, Document, RetrievedChunk, StageRank


def make_document(
    filename: str = "guide.txt", content: bytes | None = None
) -> Document:
    return Document(
        id=uuid7(),
        filename=filename,
        media_type="text/plain",
        content_hash=hashlib.sha256(content or filename.encode()).hexdigest(),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def make_chunks(document: Document, *texts: str) -> list[Chunk]:
    return [
        Chunk(
            id=uuid7(),
            document_id=document.id,
            text=text,
            position=position,
            page_start=1,
            page_end=1,
        )
        for position, text in enumerate(texts)
    ]


def make_retrieved(document: Document, *texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=chunk,
            document=document,
            dense=StageRank(rank=rank, score=1 / rank),
        )
        for rank, chunk in enumerate(make_chunks(document, *texts), start=1)
    ]


def make_docx(build: Callable[[DocxDocument], object]) -> bytes:
    document = docx.Document()
    build(document)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


# Text extraction reads only the ToUnicode CMap, so a non-embedded Type0 font with one
# 2-byte code per character round-trips any script, Cyrillic included, without a font
# file. With `encrypted`, /U matches no password, as in a user-password PDF (D45).
def make_pdf(*pages: str, encrypted: bool = False) -> bytes:
    characters = sorted({c for page in pages for c in page if c != "\n"})
    code = {c: index for index, c in enumerate(characters, start=1)}
    mappings = "\n".join(
        f"<{code[c]:04X}> <{c.encode('utf-16-be').hex()}>" for c in characters
    )
    to_unicode = (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def /CMapType 2 def\n"
        "1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
        f"{len(characters)} beginbfchar\n{mappings}\nendbfchar\n"
        "endcmap CMapName currentdict /CMap defineresource pop end end"
    )

    objects: list[str] = ["", ""]

    def add(body: str) -> int:
        objects.append(body)
        return len(objects)

    def stream(data: str) -> str:
        return f"<< /Length {len(data)} >>\nstream\n{data}\nendstream"

    catalog, pages_id = 1, 2
    cmap_id = add(stream(to_unicode))
    descriptor = add(
        "<< /Type /FontDescriptor /FontName /Helvetica /Flags 32 "
        "/FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 900 /Descent -200 "
        "/CapHeight 700 /StemV 80 >>"
    )
    cid_font = add(
        "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /Helvetica "
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        f"/FontDescriptor {descriptor} 0 R /DW 600 /CIDToGIDMap /Identity >>"
    )
    font = add(
        "<< /Type /Font /Subtype /Type0 /BaseFont /Helvetica /Encoding /Identity-H "
        f"/DescendantFonts [{cid_font} 0 R] /ToUnicode {cmap_id} 0 R >>"
    )
    kids: list[int] = []
    for text in pages:
        lines = (
            "<" + "".join(f"{code[c]:04X}" for c in line) + "> Tj T*"
            for line in text.split("\n")
        )
        content = add(stream("\n".join(["BT /F1 12 Tf 14 TL 72 720 Td", *lines, "ET"])))
        kids.append(
            add(
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"
            )
        )
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>"
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] "
        f"/Count {len(kids)} >>"
    )
    trailer = f"/Size {len(objects) + 1} /Root {catalog} 0 R"
    if encrypted:
        security = add(
            "<< /Filter /Standard /V 2 /R 3 /Length 128 /P -4 "
            f"/O <{'ab' * 32}> /U <{'cd' * 32}> >>"
        )
        trailer = (
            f"/Size {len(objects) + 1} /Root {catalog} 0 R /Encrypt {security} 0 R "
            f"/ID [<{'00' * 16}> <{'00' * 16}>]"
        )

    pdf = bytearray(b"%PDF-1.7\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n{body}\nendobj\n".encode("ascii")
    xref = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    pdf += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode("ascii")
    pdf += f"trailer\n<< {trailer} >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    return bytes(pdf)
