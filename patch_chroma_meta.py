import os
import re

import chromadb

# ─── Configuration ────────────────────────────────────────────────────────────
CHROMA_PATH = "./.chroma"
COLLECTION_NAME = "rag-chroma"
BATCH_SIZE = 500

# Minimum digit length to be considered a machine ID (avoids short numbers).
# "1005808" → 7 digits  ✓     "449" → 3 digits  ✗
MIN_ID_LENGTH = 5
# ──────────────────────────────────────────────────────────────────────────────


def machine_id_from_source(source: str) -> str | None:
    """Extract leading numeric machine ID from a source path, e.g.:
      '.../docs/1005808 07 DGUV.pdf'  →  '1005808'
      '.../docs/FB0010.01 ...'         →  None
    """
    fname = os.path.basename(source)
    m = re.match(r"^(\d{" + str(MIN_ID_LENGTH) + r",})", fname)
    return m.group(1) if m else None


def main() -> None:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION_NAME)

    results = col.get(include=["metadatas"])
    ids: list[str] = results["ids"]
    metas: list[dict] = results["metadatas"]

    print(f"Total chunks in collection: {len(ids)}")

    to_update_ids: list[str] = []
    to_update_metas: list[dict] = []

    for chunk_id, meta in zip(ids, metas):
        machine_id = machine_id_from_source(meta.get("source", ""))
        if not machine_id:
            continue  # no machine ID in filename → skip

        changed = False
        new_meta = dict(meta)

        # 1) Inject machine_id into entity_names (comma-separated, existing behaviour)
        existing: str = meta.get("entity_names", "")
        if machine_id.lower() not in existing.lower():
            new_meta["entity_names"] = (existing.rstrip(",") + "," + machine_id).lstrip(",")
            changed = True

        # 2) Write a dedicated scalar 'machine_id' field for $eq filtering
        if meta.get("machine_id", "") != machine_id:
            new_meta["machine_id"] = machine_id
            changed = True

        if changed:
            to_update_ids.append(chunk_id)
            to_update_metas.append(new_meta)

    print(f"Chunks already up-to-date: {len(ids) - len(to_update_ids)}")
    print(f"Chunks needing update:      {len(to_update_ids)}")

    if not to_update_ids:
        print("Nothing to do.")
        return

    for i in range(0, len(to_update_ids), BATCH_SIZE):
        batch_ids = to_update_ids[i : i + BATCH_SIZE]
        batch_metas = to_update_metas[i : i + BATCH_SIZE]
        col.update(ids=batch_ids, metadatas=batch_metas)
        print(f"  Updated {min(i + BATCH_SIZE, len(to_update_ids))}/{len(to_update_ids)}")

    print("Done.\n")

    # ── Spot-check ────────────────────────────────────────────────────────────
    print("Spot-check (one chunk per file, files with machine ID only):")
    verify = col.get(include=["metadatas"])
    seen: set[str] = set()
    for m in verify["metadatas"]:
        fname = os.path.basename(m.get("source", ""))
        if fname in seen:
            continue
        if machine_id_from_source(m.get("source", "")):
            seen.add(fname)
            print(f"  [{fname}]")
            print(f"    entity_names: {m.get('entity_names', '')[:120]}")


if __name__ == "__main__":
    main()
