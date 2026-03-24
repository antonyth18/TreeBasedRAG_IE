import { Upload, Plus, MessageSquare } from "lucide-react";
import { useEffect, useState, useRef } from "react";
import { fetchDocuments, uploadDocument } from "../../lib/api";

interface SidebarProps {
  selectedDocId: string | null;
  onSelectDoc: (id: string) => void;
}

interface DocumentInfo {
  id: string;
  filename: string;
  status: string;
  upload_time: string;
}

export function Sidebar({ selectedDocId, onSelectDoc }: SidebarProps) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const data = await fetchDocuments();
      setDocuments(data);
      // Auto-select first document if none selected
      if (data.length > 0 && !selectedDocId) {
        onSelectDoc(data[0].id);
      }
    } catch (error) {
      console.error("Error loading documents:", error);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const res = await uploadDocument(file);
      await loadDocuments();
      onSelectDoc(res.document_id);
    } catch (error) {
      console.error("Error uploading document:", error);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="flex h-full w-64 flex-col bg-[#333333] p-3">
      {/* Upload Document Button */}
      <button 
        onClick={() => fileInputRef.current?.click()}
        disabled={isUploading}
        className="mb-3 flex w-full items-center justify-center gap-2 rounded-lg bg-white/10 px-3 py-2.5 text-sm font-medium text-white transition-all hover:bg-white/15 disabled:opacity-50"
      >
        <Plus className="h-4 w-4" />
        {isUploading ? "Uploading..." : "New Chat"}
      </button>
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        className="hidden" 
        accept=".pdf,.txt,.md"
      />

      {/* Upload Area */}
      <div 
        onClick={() => fileInputRef.current?.click()}
        className="mb-4 cursor-pointer rounded-lg border border-white/10 bg-white/5 p-4 text-center hover:bg-white/10"
      >
        <Upload className="mx-auto mb-2 h-6 w-6 text-white/40" />
        <p className="mb-1 text-xs text-white/60">Upload Document</p>
        <p className="text-xs text-white/40">PDF, TXT, MD</p>
      </div>

      {/* Conversations List */}
      <div className="mb-3 flex items-center justify-between px-1">
        <h3 className="text-xs font-medium text-white/60">Documents</h3>
      </div>

      <div className="flex-1 space-y-1 overflow-auto">
        {documents.map((doc) => (
          <button
            key={doc.id}
            onClick={() => onSelectDoc(doc.id)}
            className={`group w-full rounded-lg px-3 py-2.5 text-left transition-all ${
              selectedDocId === doc.id
                ? "bg-white/10"
                : "hover:bg-white/5"
            }`}
          >
            <div className="mb-1 flex items-center gap-2">
              <MessageSquare className="h-3.5 w-3.5 flex-shrink-0 text-white/60" />
              <p className="flex-1 truncate text-sm text-white">
                {doc.filename}
              </p>
            </div>
            <p className="text-xs text-white/40">{doc.status}</p>
          </button>
        ))}
      </div>
    </div>
  );
}