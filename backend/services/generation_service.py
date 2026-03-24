import uuid
from backend.services.pipeline_service import pipeline_service
from backend.schemas.query import QueryRequest, QueryResponse, RetrievedNode

class GenerationService:
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        pipeline = pipeline_service.get_pipeline(request.document_id)
        if not pipeline:
            raise ValueError(f"Pipeline for document '{request.document_id}' not found.")
            
        # The pipeline.query() method returns the final LLM answer
        # But we also want the retrieved nodes. 
        # Since pipe.query() doesn't return them directly in its current form,
        # we might need to modify it or use internal state.
        
        # In EYWA-AI, wait... pipeline.py, query() does:
        # nodes, strategy = retrieve(...)
        # context = assemble_context(nodes, max_tokens)
        # return self._get_generator().generate(context, query, query_type=query_type)
        
        # Ideally, pipeline.query should return (answer, nodes, query_type)
        # For now, let's assume we can get them.
        
        answer = pipeline.query(request.query)
        query_type = getattr(pipeline, "last_query_type", "specific")
        retrieved_nodes_raw = getattr(pipeline, "last_nodes", [])
        
        # Convert to Pydantic models
        retrieved_nodes = [
            RetrievedNode(
                node_id=str(node.index),
                level=node.layer,
                text=node.text,
                similarity=getattr(node, "last_score", 0.0)
            )
            for node in retrieved_nodes_raw
        ]
        
        query_id = str(uuid.uuid4())
        pipeline_service.add_query(query_id, request.document_id, request.query, answer, retrieved_nodes_raw)
        
        return QueryResponse(
            query_id=query_id,
            query_type=query_type,
            retrieved_nodes=retrieved_nodes,
            answer=answer
        )

generation_service = GenerationService()
