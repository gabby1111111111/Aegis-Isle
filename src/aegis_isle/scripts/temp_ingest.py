import os

file_path = "E:/SillyTaven/SillyTavern/data/default-user/chats/邹峥1/邹峥 - 2026-02-27@17h23m32s.jsonl"
character = "邹峥"
world_line = None

# Ensure the output directory exists
index_path = f"data/vectorstore/st_memory/{character}.index"
os.makedirs(index_path, exist_ok=True)

import aegis_isle.rag.st_memory_manager

manager = aegis_isle.rag.st_memory_manager.STMemoryManager()
from aegis_isle.scripts.ingest_st_chats import logger, parse_st_chat_log

logger.info(f"Starting ingestion process for {file_path}")
chunks = parse_st_chat_log(file_path, character, world_line, 1)

if chunks:
    manager.ingest_chunks(chunks, character, world_line)
    logger.info("Ingestion complete.")
else:
    logger.warning("No chunks were created. Aborting ingestion.")
