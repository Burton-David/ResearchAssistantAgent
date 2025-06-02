#!/usr/bin/env python3
"""
Migration tool for converting pickle-based vector stores to JSON format.
"""

import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
import logging

from .vector_store import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_vector_store(store_path: Path) -> None:
    """Migrate vector store from pickle to JSON format."""
    
    # Check paths
    pickle_path = store_path / "documents.pkl"
    json_path = store_path / "documents.json"
    
    if not pickle_path.exists():
        logger.error(f"No pickle file found at {pickle_path}")
        return
        
    if json_path.exists():
        logger.warning(f"JSON file already exists at {json_path}")
        if input("Overwrite? (y/n): ").lower() != "y":
            return
            
    # Load pickle data
    logger.info(f"Loading pickle data from {pickle_path}")
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load pickle file: {e}")
        return
        
    # Convert documents
    logger.info("Converting documents to JSON format...")
    documents_data = {
        "documents": {},
        "id_to_idx": data.get("id_to_idx", {}),
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "migrated_from": "pickle"
    }
    
    # Convert each document
    for idx, doc in data.get("documents", {}).items():
        if hasattr(doc, "to_dict"):
            # New Document class
            documents_data["documents"][str(idx)] = doc.to_dict()
        elif hasattr(doc, "page_content"):
            # Legacy format
            documents_data["documents"][str(idx)] = {
                "id": getattr(doc, "id", f"doc_{idx}"),
                "text": getattr(doc, "page_content", getattr(doc, "text", "")),
                "metadata": getattr(doc, "metadata", {}),
                "embedding": getattr(doc, "embedding", None)
            }
        else:
            logger.warning(f"Unknown document format for index {idx}")
            
    # Save JSON data
    logger.info(f"Saving JSON data to {json_path}")
    try:
        with open(json_path, "w") as f:
            json.dump(documents_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save JSON file: {e}")
        return
        
    # Backup pickle file
    backup_path = pickle_path.with_suffix(".pkl.backup")
    logger.info(f"Creating backup at {backup_path}")
    pickle_path.rename(backup_path)
    
    logger.info("Migration completed successfully!")
    logger.info(f"Original pickle file backed up to: {backup_path}")
    logger.info(f"New JSON file created at: {json_path}")


def main():
    """Main entry point for migration tool."""
    if len(sys.argv) != 2:
        print("Usage: python -m research_assistant.migrate_vector_store <store_path>")
        sys.exit(1)
        
    store_path = Path(sys.argv[1])
    if not store_path.exists():
        logger.error(f"Path does not exist: {store_path}")
        sys.exit(1)
        
    migrate_vector_store(store_path)


if __name__ == "__main__":
    main()