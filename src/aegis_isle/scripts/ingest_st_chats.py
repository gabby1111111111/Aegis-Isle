import os
import json
import argparse
import logging
from typing import List

from aegis_isle.rag.st_memory import ChatChunk
from aegis_isle.rag.st_memory_manager import STMemoryManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import re

def clean_st_text(text: str) -> str:
    # 1. Strip HTML comments, prologue, and bracket tags
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<prologue>.*?</prologue>', '', text, flags=re.DOTALL)
    text = re.sub(r'\[.*?\]', '', text)
    
    # 2. Strip UI and meta-game text tokens instead of truncating everything after them
    cutoff_pattern = r'(当前bgm:|⋯♡⋯|```html|```mermaid|𐙚₊˚|☆₊⁺|𓋫 𓏴𓏴|【小剧场|剧情分支).*?(\n|$)'
    text = re.sub(cutoff_pattern, '', text, flags=re.IGNORECASE)
        
    # 3. Strip specific tag markers like <aurora_time>, <content>, <li> but keep their inner text
    text = re.sub(r'</?(?:content|aurora_time|li)[^>]*>', '', text)
    
    # Clean up empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

def parse_st_chat_log(file_path: str, character_name: str, world_line: str = None, chunk_size: int = 1) -> List[ChatChunk]:
    """
    Parses a SillyTavern JSONL file and returns a list of ChatChunk objects.
    Using improved chunking: each AI response is a separate chunk, prefixed by the preceding user message.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    messages = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if "chat_metadata" in data:
                    continue
                
                if "mes" in data and "is_user" in data:
                    is_user = data["is_user"]
                    name = data.get("name", "User" if is_user else character_name)
                    # Apply regex cleaning to both user and AI messages to be safe
                    content = clean_st_text(data["mes"])
                    
                    if not content:
                        continue
                        
                    timestamp = data.get("send_date", None)
                    
                    messages.append({
                        "role": "user" if is_user else "char",
                        "name": name,
                        "content": content,
                        "timestamp": timestamp
                    })
            except json.JSONDecodeError:
                continue

    logger.info(f"Extracted {len(messages)} cleaned messages from {file_path}")
    
    chunks = []
    
    # Improved Chunking: Pair User message with AI message
    for i in range(len(messages)):
        msg = messages[i]
        if msg["role"] == "char":
            # Start gathering context for this AI response
            start_time = None
            
            # Look backwards for the immediately preceding user message(s)
            base_context = ""
            if i > 0 and messages[i-1]["role"] == "user":
                prev_msg = messages[i-1]
                base_context = f"{prev_msg['name']}: {prev_msg['content']}\n\n"
                start_time = prev_msg.get("timestamp")
            
            if not start_time:
                start_time = msg.get("timestamp")
            end_time = msg.get("timestamp")
            
            ai_text = msg['content']
            
            # If AI message is very long, break it into smaller sub-chunks by paragraph
            # but keep the user context attached to each sub-chunk
            if len(ai_text) > 800:
                paragraphs = ai_text.split('\n\n')
                current_sub_chunk_text = base_context + f"{msg['name']}: "
                
                for para in paragraphs:
                    if not para.strip():
                        continue
                        
                    # 1000 chars is comfortably within the ~512 token FAISS limit for most embeddings
                    if len(current_sub_chunk_text) + len(para) > 1000 and len(current_sub_chunk_text) > len(base_context + f"{msg['name']}: "):
                        chunks.append(ChatChunk(
                            text=current_sub_chunk_text.strip(),
                            character_name=character_name,
                            chat_file=os.path.basename(file_path),
                            start_time=start_time,
                            end_time=end_time,
                            world_line=world_line
                        ))
                        # Start next sub-chunk with context again
                        current_sub_chunk_text = base_context + f"{msg['name']}: " + para + "\n\n"
                    else:
                        current_sub_chunk_text += para + "\n\n"
                
                # Add any remaining text as a chunk
                if len(current_sub_chunk_text) > len(base_context + f"{msg['name']}: "):
                     chunks.append(ChatChunk(
                        text=current_sub_chunk_text.strip(),
                        character_name=character_name,
                        chat_file=os.path.basename(file_path),
                        start_time=start_time,
                        end_time=end_time,
                        world_line=world_line
                    ))
            else:
                # Normal short/medium message chunking
                chunk_text = base_context + f"{msg['name']}: {msg['content']}"
                chunk = ChatChunk(
                    text=chunk_text.strip(),
                    character_name=character_name,
                    chat_file=os.path.basename(file_path),
                    start_time=start_time,
                    end_time=end_time,
                    world_line=world_line
                )
                chunks.append(chunk)

    logger.info(f"Created {len(chunks)} contextual chunks from {file_path}")
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Ingest SillyTavern JSONL chat logs into Aegis-Isle ST Memory FAISS Index.")
    parser.add_argument("file_path", type=str, help="Path to the .jsonl chat file")
    parser.add_argument("--character", type=str, required=True, help="The name of the character (e.g., 邹峥)")
    parser.add_argument("--world-line", type=str, default=None, help="Optional world line namespace to group chats together (e.g., AIDom项目宇宙)")
    parser.add_argument("--chunk-size", type=int, default=6, help="Number of conversational turns per chunk")
    
    args = parser.parse_args()
    
    manager = STMemoryManager()
    
    logger.info(f"Starting ingestion process for {args.file_path}")
    chunks = parse_st_chat_log(args.file_path, args.character, args.world_line, args.chunk_size)
    
    if chunks:
        manager.ingest_chunks(chunks, args.character, args.world_line)
        logger.info("Ingestion complete.")
    else:
        logger.warning("No chunks were created. Aborting ingestion.")

if __name__ == "__main__":
    main()
