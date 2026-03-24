import os
import uuid
import shutil
from typing import Optional
from backend.core.config import settings
from backend.services.pipeline_service import pipeline_service

# Add parent directory to sys.path to import RaptorPipeline
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline import RaptorPipeline

class IngestionService:
    def __init__(self):
        # Ensure data and trees directories exist
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        os.makedirs(settings.TREES_DIR, exist_ok=True)

    async def upload_document(self, file_content: bytes, filename: str) -> str:
        import hashlib
        # Use content hash as unique ID to avoid duplicates
        content_hash = hashlib.sha256(file_content).hexdigest()
        document_id = content_hash[:16] # Use a shorter hash for ID
        
        # Check if already in memory
        existing_doc = pipeline_service.get_document_info(document_id)
        if existing_doc and existing_doc["status"] == "completed":
            return document_id

        # Save file to data directory (use hash as filename for physical deduplication)
        # We keep the extension from the original filename if possible
        ext = os.path.splitext(filename)[1]
        file_path = os.path.join(settings.DATA_DIR, f"{document_id}{ext}")
        
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(file_content)
        
        # Initialize RaptorPipeline
        pipeline = RaptorPipeline(llm_model=settings.DEFAULT_LLM_MODEL)
        tree_path = os.path.join(settings.TREES_DIR, f"{document_id}.json")
        
        # Add to pipeline service
        pipeline_service.add_document(document_id, filename, pipeline)
        
        try:
            # Build method now handles internal loading if tree_path exists
            tree = pipeline.build(file_path, save_path=tree_path)
            
            pipeline_service.update_document_info(
                document_id,
                status="completed",
                num_chunks=len(tree.all_nodes_flat()),
                num_nodes=len(tree.all_nodes_flat())
            )
        except Exception as e:
            pipeline_service.update_document_info(document_id, status=f"failed: {str(e)}")
            
        return document_id

ingestion_service = IngestionService()
