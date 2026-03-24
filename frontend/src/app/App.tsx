import { useState } from "react";
import { Header } from "./components/Header";
import { QuerySection } from "./components/QuerySection";
import { Sidebar } from "./components/Sidebar";
import { TreeVisualization } from "./components/TreeVisualization";
import { RetrievalDetails } from "./components/RetrievalDetails";
import { GeneratedAnswer } from "./components/GeneratedAnswer";
import { queryDocument } from "../lib/api";

export default function App() {
  const [query, setQuery] = useState("");
  const [selectedDocId, setSelectedDocId] = useState<string | null>(null);
  const [queryId, setQueryId] = useState<string | null>(null);
  const [answer, setAnswer] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const handleRunQuery = async () => {
    if (!selectedDocId || !query.trim()) return;
    
    setIsLoading(true);
    setAnswer("");
    setQueryId(null);
    
    try {
      const res = await queryDocument({
        document_id: selectedDocId,
        query: query
      });
      
      setAnswer(res.answer);
      setQueryId(res.query_id);
    } catch (error) {
      console.error("Error running query:", error);
      setAnswer("Failed to generate answer. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-[#B0BEC5]/10">
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar 
          selectedDocId={selectedDocId} 
          onSelectDoc={setSelectedDocId} 
        />
        
        {/* Main Content Area */}
        <div className="flex flex-1 flex-col overflow-hidden">
          <Header />
          <QuerySection 
            query={query} 
            onQueryChange={setQuery} 
            onRunQuery={handleRunQuery} 
            isDisabled={!selectedDocId || isLoading}
          />
          
          <div className="flex-1 overflow-hidden bg-[#B0BEC5]/10">
            <div className="mx-auto h-full max-w-[1600px] px-8 py-8">
              <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-3">
                {/* Left Panel - Tree Visualization */}
                <div className="h-full overflow-hidden">
                  <TreeVisualization documentId={selectedDocId} queryId={queryId} />
                </div>

                {/* Middle Panel - Retrieval Details */}
                <div className="h-full overflow-hidden">
                  <RetrievalDetails queryId={queryId} />
                </div>

                {/* Right Panel - Generated Answer */}
                <div className="h-full overflow-hidden">
                  <GeneratedAnswer answer={answer} isLoading={isLoading} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}