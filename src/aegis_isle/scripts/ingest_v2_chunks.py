"""
将 debug/chunks/ 下的 *_sub_chunks.jsonl 灌入 FAISS
"""
import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.aegis_isle.rag.st_memory_manager import memory_manager
from src.aegis_isle.rag.st_memory import ChatChunk
from src.aegis_isle.core.logging import logger

def ingest_debug_sub_chunks(debug_dir="debug/chunks"):
    debug_path = Path(debug_dir)
    if not debug_path.exists():
        logger.error(f"Directory {debug_dir} does not exist.")
        return

    sub_chunk_files = list(debug_path.glob("*_sub_chunks.jsonl"))
    if not sub_chunk_files:
        logger.warning(f"No *_sub_chunks.jsonl found in {debug_dir}")
        return
        
    logger.info(f"Found {len(sub_chunk_files)} sub_chunks files for ingestion.")

    for fpath in sub_chunk_files:
        logger.info(f"Processing {fpath.name}...")
        chunks_to_ingest = []
        
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    # Convert to ChatChunk expected by st_memory_manager
                    chunk = ChatChunk(
                        text=data.get("text", ""),
                        character_name=data.get("character_name", ""),
                        chat_file=fpath.name,
                        start_time=data.get("start_time", None),
                        end_time=data.get("end_time", None),
                        world_line=data.get("universe_id", ""),
                        parent_chunk_id=data.get("parent_chunk_id", None)
                    )
                    chunks_to_ingest.append(chunk)
                except Exception as e:
                    logger.error(f"Error parsing line in {fpath.name}: {e}")
        
        if chunks_to_ingest:
            # Group by universe_id for ingestion
            by_universe = {}
            for c in chunks_to_ingest:
                key = (c.character_name, c.world_line)
                if key not in by_universe:
                    by_universe[key] = []
                by_universe[key].append(c)
                
            for (char, universe), char_chunks in by_universe.items():
                logger.info(f"Ingesting {len(char_chunks)} chunks for {char} in universe {universe}")
                memory_manager.ingest_chunks(char_chunks, character_name=char, world_line=universe)
        else:
            logger.warning(f"No valid chunks parsed from {fpath.name}")

if __name__ == "__main__":
    ingest_debug_sub_chunks()
