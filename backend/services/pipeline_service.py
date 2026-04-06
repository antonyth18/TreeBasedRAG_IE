import sys
import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

# Add parent directory to sys.path to import RaptorPipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline import RaptorPipeline

class PipelineService:
    def __init__(self):
        # document_id -> { "pipeline": RaptorPipeline, "info": DocumentInfo }
        self.documents: Dict[str, Dict[str, Any]] = {}
        # query_id -> { "query": str, "response": QueryResponse, "nodes": List[RaptorNode] }
        self.queries: Dict[str, Dict[str, Any]] = {}
        # conversation_id -> List[Message]
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load existing documents on startup
        self.load_existing_documents()

    def load_existing_documents(self):
        from backend.core.config import settings
        if not os.path.exists(settings.TREES_DIR):
            return
            
        # The tree_serializer saves trees as DIRECTORIES ending in .json/
        # inside which are tree.json and embeddings.npz
        for entry in os.listdir(settings.TREES_DIR):
            entry_path = os.path.join(settings.TREES_DIR, entry)
            if os.path.isdir(entry_path) and entry.endswith(".json"):
                document_id = entry.replace(".json", "")
                
                # Check if we have a corresponding data file
                data_file = "Unknown Document"
                if os.path.exists(settings.DATA_DIR):
                    for df in os.listdir(settings.DATA_DIR):
                        if df.startswith(document_id):
                            data_file = df
                            break
                            
                try:
                    pipeline = RaptorPipeline(
                        llm_model=settings.DEFAULT_LLM_MODEL,
                        summary_model=settings.SUMMARY_MODEL,
                        summary_max_tokens=settings.SUMMARY_MAX_TOKENS,
                        summary_max_retries=settings.SUMMARY_MAX_RETRIES,
                        summary_retry_delay=settings.SUMMARY_RETRY_DELAY,
                        summary_verify_faithfulness=settings.SUMMARY_VERIFY_FAITHFULNESS,
                        summary_max_verification_retries=settings.SUMMARY_MAX_VERIFICATION_RETRIES,
                        enable_web_search=settings.ENABLE_WEB_SEARCH,
                        web_search_threshold=settings.WEB_SEARCH_THRESHOLD,
                        web_search_n_results=settings.WEB_SEARCH_N_RESULTS,
                    )
                    # The path to load is the directory itself
                    pipeline.load(entry_path)
                    
                    # Add to in-memory state
                    # data_file could be "e71e4798d0040713.pdf" or "e71e4798d0040713_datafile1.pdf"
                    filename_display = data_file
                    if data_file.startswith(document_id):
                        # Remove hash and potential underscore
                        filename_display = data_file[len(document_id):].lstrip("_")
                        if not filename_display: # Case where filename was just the hash
                            filename_display = data_file

                    self.add_document(
                        document_id=document_id,
                        filename=filename_display,
                        pipeline=pipeline,
                        status="completed"
                    )
                    
                    # Update node counts
                    if pipeline._tree:
                        self.update_document_info(
                            document_id,
                            num_chunks=len(pipeline._tree.all_nodes_flat()),
                            num_nodes=len(pipeline._tree.all_nodes_flat())
                        )
                    print(f"Auto-loaded document {document_id} from disk.")
                except Exception as e:
                    print(f"Failed to auto-load document {document_id}: {e}")

    def get_pipeline(self, document_id: str) -> Optional[RaptorPipeline]:
        doc = self.documents.get(document_id)
        return doc["pipeline"] if doc else None

    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        for doc in self.documents.values():
            if doc["info"]["filename"] == filename:
                return doc
        return None

    def add_document(self, document_id: str, filename: str, pipeline: RaptorPipeline, status: str = "processing"):
        if document_id in self.documents:
            # Update existing document if needed, but don't reset upload_time
            self.documents[document_id]["pipeline"] = pipeline
            self.documents[document_id]["info"]["status"] = status
            return

        self.documents[document_id] = {
            "pipeline": pipeline,
            "info": {
                "id": document_id,
                "filename": filename,
                "status": status,
                "upload_time": datetime.now(),
                "num_chunks": 0,
                "num_nodes": 0
            }
        }

    def update_document_info(self, document_id: str, **kwargs):
        if document_id in self.documents:
            self.documents[document_id]["info"].update(kwargs)

    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.get(document_id, {}).get("info")

    def list_documents(self) -> List[Dict[str, Any]]:
        return [doc["info"] for doc in self.documents.values()]

    def add_query(self, query_id: str, document_id: str, query: str, response: Any, nodes: List[Any]):
        self.queries[query_id] = {
            "document_id": document_id,
            "query": query,
            "response": response,
            "nodes": nodes
        }

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        return self.queries.get(query_id)

pipeline_service = PipelineService()
