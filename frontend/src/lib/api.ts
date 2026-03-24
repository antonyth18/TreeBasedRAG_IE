const API_BASE = "http://localhost:8000/api/v1";

async function handleResponse(response: Response) {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function fetchDocuments() {
  return fetch(`${API_BASE}/documents`).then(handleResponse);
}

export async function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  return fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    body: formData,
  }).then(handleResponse);
}

export async function queryDocument(payload: { document_id: string, query: string, conversation_id?: string }) {
  return fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  }).then(handleResponse);
}

export async function getRetrievedNodes(queryId: string) {
  return fetch(`${API_BASE}/query/${queryId}/retrieved-nodes`)
    .then(handleResponse);
}

export async function getTreeSummary(documentId: string) {
  return fetch(`${API_BASE}/documents/${documentId}/tree/summary`)
    .then(handleResponse);
}

export async function getRetrievalSummary(queryId: string) {
  return fetch(`${API_BASE}/query/${queryId}/retrieval-summary`)
    .then(handleResponse);
}
