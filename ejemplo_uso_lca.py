#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exporta un Excel con tablas a UN SOLO documento de texto para RAG.
- Detecta automáticamente la fila de cabecera.
- Extrae metadatos (pares clave-valor) por encima de la cabecera (col A/B).
- Rellena valores de celdas combinadas.
- Convierte la tabla en registros tipo KV (recomendado) y/o Markdown.
- Une todo en un único archivo: output.txt (o .md).
"""

import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import openpyxl


# -----------------------------
# Utilidades
# -----------------------------
def norm_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return re.sub(r"\s+", " ", x.strip())
    return str(x)


def is_empty_row(values: List[Any]) -> bool:
    return all(norm_str(v) == "" for v in values)


def build_merged_lookup(ws) -> Dict[Tuple[int, int], Tuple[int, int]]:
    lookup = {}
    for merged_range in ws.merged_cells.ranges:
        min_row, min_col, max_row, max_col = (
            merged_range.min_row,
            merged_range.min_col,
            merged_range.max_row,
            merged_range.max_col,
        )
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                lookup[(r, c)] = (min_row, min_col)
    return lookup


def cell_value(ws, merged_lookup: Dict[Tuple[int, int], Tuple[int, int]], r: int, c: int) -> Any:
    v = ws.cell(r, c).value
    if v is None and (r, c) in merged_lookup:
        tr, tc = merged_lookup[(r, c)]
        v = ws.cell(tr, tc).value
    return v


def get_row_values(ws, merged_lookup, r: int, max_col: int) -> List[Any]:
    return [cell_value(ws, merged_lookup, r, c) for c in range(1, max_col + 1)]


# -----------------------------
# Detección de cabecera (heurística)
# -----------------------------
def looks_like_header(row_vals: List[Any]) -> bool:
    non_empty = [v for v in row_vals if norm_str(v) != ""]
    if len(non_empty) < 3:
        return False

    str_count = sum(isinstance(v, str) for v in non_empty)
    if str_count / len(non_empty) < 0.7:
        return False

    text = " | ".join(norm_str(v) for v in non_empty).upper()
    header_tokens = [
        "PROVEEDOR", "CATEGOR", "SUBCATEGOR", "TIPO", "CANTIDAD",
        "UNIDAD", "DESCRIP", "FACTOR", "EMISION", "HUELLA"
    ]
    bonus = sum(1 for t in header_tokens if t in text)
    return bonus >= 1


def find_header_row(ws, merged_lookup, max_scan_rows: int = 100) -> Optional[int]:
    max_col = ws.max_column
    scan_limit = min(ws.max_row, max_scan_rows)
    for r in range(1, scan_limit + 1):
        row_vals = get_row_values(ws, merged_lookup, r, max_col)
        if looks_like_header(row_vals):
            return r
    return None


# -----------------------------
# Metadata encima de la cabecera (A/B)
# -----------------------------
def extract_metadata(ws, merged_lookup, header_row: int, max_pairs_rows: int = 60) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    start = max(1, header_row - max_pairs_rows)
    for r in range(start, header_row):
        k = norm_str(cell_value(ws, merged_lookup, r, 1))
        v = norm_str(cell_value(ws, merged_lookup, r, 2))
        if k and v and len(k) <= 80:
            meta[k] = v
    return meta


# -----------------------------
# Lectura de la tabla
# -----------------------------
def read_table(ws, merged_lookup, header_row: int) -> Tuple[List[str], List[List[str]]]:
    max_col = ws.max_column
    header_vals = get_row_values(ws, merged_lookup, header_row, max_col)
    headers = [norm_str(v) for v in header_vals]

    # recortar columnas vacías al final
    last_col = 0
    for i, h in enumerate(headers, start=1):
        if h != "":
            last_col = i
    headers = headers[:last_col]

    rows: List[List[str]] = []
    empty_streak = 0
    for r in range(header_row + 1, ws.max_row + 1):
        vals = [norm_str(cell_value(ws, merged_lookup, r, c)) for c in range(1, last_col + 1)]
        if is_empty_row(vals):
            empty_streak += 1
            # tolera 1-2 filas vacías, luego corta (muchos excels usan vacíos de separación)
            if empty_streak >= 2:
                break
            continue
        empty_streak = 0
        rows.append(vals)

    return headers, rows


# -----------------------------
# Formatos de salida
# -----------------------------
def to_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    def esc(x: str) -> str:
        return x.replace("|", "\\|")

    out = []
    out.append("| " + " | ".join(esc(h) if h else "(vacío)" for h in headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(esc(v) for v in row) + " |")
    return "\n".join(out)


def to_kv_records(headers: List[str], rows: List[List[str]], start_index: int = 1) -> str:
    out = []
    for i, row in enumerate(rows, start=start_index):
        out.append(f"Registro {i}:")
        for h, v in zip(headers, row):
            if v != "":
                out.append(f"- {h}: {v}")
        out.append("")  # blank line between records
    return "\n".join(out).strip()


def chunk_rows(rows: List[List[str]], chunk_size: int) -> List[List[List[str]]]:
    if chunk_size <= 0:
        return [rows]
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Excel -> UN solo documento para RAG (KV/Markdown).")
    ap.add_argument("-i", "--input", required=True, help="Ruta del archivo .xlsx")
    ap.add_argument("-o", "--output", required=True, help="Ruta del archivo de salida (ej: rag_doc.txt)")
    ap.add_argument("--format", choices=["kv", "markdown", "both"], default="kv",
                    help="Formato de exportación dentro del documento.")
    ap.add_argument("--chunk-rows", type=int, default=50,
                    help="Filas por bloque para evitar secciones gigantes. 0=sin chunk.")
    ap.add_argument("--max-scan-rows", type=int, default=100,
                    help="Filas máximas para detectar cabecera.")
    args = ap.parse_args()

    wb = openpyxl.load_workbook(args.input, data_only=True)

    parts: List[str] = []
    parts.append(f"DOCUMENTO RAG (origen: {os.path.basename(args.input)})")
    parts.append("=" * 80)
    parts.append("")

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        merged_lookup = build_merged_lookup(ws)

        header_row = find_header_row(ws, merged_lookup, max_scan_rows=args.max_scan_rows)

        parts.append("#" * 80)
        parts.append(f"SHEET: {sheet_name}")
        parts.append("#" * 80)
        parts.append("")

        if header_row is None:
            # fallback: volcar filas no vacías como texto lineal
            parts.append("NOTA: No se detectó cabecera de tabla. Volcado RAW (filas no vacías).")
            parts.append("")
            for r in range(1, ws.max_row + 1):
                row_vals = [norm_str(cell_value(ws, merged_lookup, r, c)) for c in range(1, ws.max_column + 1)]
                if not is_empty_row(row_vals):
                    line = " | ".join(v for v in row_vals if v)
                    if line.strip():
                        parts.append(line)
            parts.append("")
            continue

        meta = extract_metadata(ws, merged_lookup, header_row)
        headers, rows = read_table(ws, merged_lookup, header_row)

        parts.append(f"Tabla detectada: fila cabecera = {header_row}")
        parts.append("")

        if meta:
            parts.append("METADATA:")
            for k, v in meta.items():
                parts.append(f"- {k}: {v}")
            parts.append("")

        if not headers or not rows:
            parts.append("NOTA: Tabla vacía o sin filas.")
            parts.append("")
            continue

        # Chunking dentro del mismo documento (pero sigue siendo un solo archivo)
        chunks = chunk_rows(rows, args.chunk_rows)
        for idx, chunk in enumerate(chunks, start=1):
            parts.append("-" * 80)
            parts.append(f"SECCIÓN TABLA | Chunk {idx}/{len(chunks)} | filas={len(chunk)}")
            parts.append("-" * 80)
            parts.append("")

            if args.format in ("kv", "both"):
                parts.append("FORMATO: KV (recomendado para RAG)")
                parts.append(to_kv_records(headers, chunk, start_index=(idx - 1) * (args.chunk_rows if args.chunk_rows > 0 else len(rows)) + 1))
                parts.append("")

            if args.format in ("markdown", "both"):
                parts.append("FORMATO: Markdown Table")
                parts.append(to_markdown_table(headers, chunk))
                parts.append("")

        parts.append("")

    # escribir salida única
    out_text = "\n".join(parts).rstrip() + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out_text)

    print(f"OK: generado un único documento -> {args.output}")


if __name__ == "__main__":
    main()