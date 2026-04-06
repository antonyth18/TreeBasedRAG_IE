import { useState } from "react";
import { Header } from "./components/Header";
import { QuerySection } from "./components/QuerySection";
import { Sidebar } from "./components/Sidebar";
import { TreeModal } from "./components/TreeModal";
import { NodesModal } from "./components/NodesModal";
import { queryDocument } from "../lib/api";
import { MessageCircle, BarChart3 } from "lucide-react";

interface Message {
  id: string;
  query: string;
  answer: string;
  queryId: string;
  timestamp: Date;
}

export default function App() {
  const [query, setQuery] = useState("");
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [showTreeModal, setShowTreeModal] = useState(false);
  const [showNodesModal, setShowNodesModal] = useState(false);
  const [selectedTreeQueryId, setSelectedTreeQueryId] = useState<string | null>(null);
  const [selectedNodesQueryId, setSelectedNodesQueryId] = useState<string | null>(null);

  const handleRunQuery = async () => {
    if (!selectedDocId || !query.trim()) return;

    setIsLoading(true);

    try {
      const res = await queryDocument({
        document_id: selectedDocId,
        query,
      });

      const newMessage: Message = {
        id: Date.now().toString(),
        query,
        answer: res.answer,
        queryId: res.query_id,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, newMessage]);
      setQuery("");
    } catch (error) {
      console.error("Error running query:", error);
      alert("Failed to generate answer. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-background">
      <Sidebar selectedDocId={selectedDocId} onSelectDoc={setSelectedDocId} />

      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />

        <div className="flex-1 overflow-auto">
          <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 md:px-8 lg:px-0">
            {messages.length === 0 ? (
              <div className="flex h-80 flex-col items-center justify-center gap-4 text-center">
                <div className="flex items-center justify-center gap-3">
                  <img src="/logo.png" alt="EywaAI logo" className="h-12 w-12 rounded-xl object-cover" />
                  <h2 className="text-3xl font-semibold text-foreground">EywaAI</h2>
                </div>
                <p className="text-muted-foreground">Upload a document and ask questions about it</p>
              </div>
            ) : (
              <div className="space-y-6">
                {messages.map((msg) => (
                  <div key={msg.id} className="space-y-4">
                    <div className="flex justify-end">
                      <div className="max-w-2xl rounded-lg bg-primary/20 px-4 py-3 text-foreground">{msg.query}</div>
                    </div>

                    <div className="flex justify-start">
                      <div className="max-w-2xl space-y-3">
                        <div className="rounded-lg bg-muted px-4 py-3 text-foreground">{msg.answer}</div>

                        <div className="flex gap-2 pt-2">
                          <button
                            onClick={() => {
                              setSelectedTreeQueryId(msg.queryId);
                              setShowTreeModal(true);
                            }}
                            className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground transition-colors hover:bg-muted"
                          >
                            <BarChart3 className="h-4 w-4" />
                            Tree Structure
                          </button>
                          <button
                            onClick={() => {
                              setSelectedNodesQueryId(msg.queryId);
                              setShowNodesModal(true);
                            }}
                            className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground transition-colors hover:bg-muted"
                          >
                            <MessageCircle className="h-4 w-4" />
                            Retrieved Nodes
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex justify-start">
                    <div className="flex items-center gap-2 rounded-lg bg-muted px-4 py-3">
                      <div className="flex gap-1">
                        <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary" style={{ animationDelay: "0ms" }} />
                        <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary" style={{ animationDelay: "150ms" }} />
                        <div className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary" style={{ animationDelay: "300ms" }} />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-border bg-background/80 backdrop-blur-sm">
          <div className="mx-auto max-w-4xl px-4 py-4 sm:px-6 md:px-8 lg:px-0">
            <QuerySection
              query={query}
              onQueryChange={setQuery}
              onRunQuery={handleRunQuery}
              isDisabled={!selectedDocId || isLoading}
            />
          </div>
        </div>
      </div>

      {showTreeModal && selectedTreeQueryId && (
        <TreeModal documentId={selectedDocId} onClose={() => setShowTreeModal(false)} />
      )}

      {showNodesModal && selectedNodesQueryId && (
        <NodesModal queryId={selectedNodesQueryId} onClose={() => setShowNodesModal(false)} />
      )}
    </div>
  );
}
