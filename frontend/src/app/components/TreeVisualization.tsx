import { FolderTree, Layers, Activity } from "lucide-react";
import { useEffect, useState } from "react";
import { getTreeSummary, getRetrievalSummary, getDocumentStatus } from "../../lib/api";

interface TreeLevelSummary {
  level: number;
  num_clusters: number;
}

interface TreeSummary {
  document_id: string;
  levels: TreeLevelSummary[];
  total_nodes: number;
  max_depth: number;
}

interface TreeVisualizationProps {
  documentId: string | null;
  queryId?: string | null;
}

export function TreeVisualization({ documentId, queryId }: TreeVisualizationProps) {
  const [summary, setSummary] = useState<TreeSummary | null>(null);
  const [retrievalStats, setRetrievalStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [buildStatus, setBuildStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!documentId) {
      setSummary(null);
      setBuildStatus(null);
      return;
    }

    let isMounted = true;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const loadTreeData = async () => {
      setIsLoading(true);
      try {
        const data = await getTreeSummary(documentId);
        if (!isMounted) return;
        setSummary(data);
        setBuildStatus("completed");
        if (intervalId) {
          clearInterval(intervalId);
          intervalId = null;
        }
      } catch {
        try {
          const status = await getDocumentStatus(documentId);
          if (!isMounted) return;
          setBuildStatus(status.status || "processing");
          if (typeof status.status === "string" && status.status.toLowerCase().startsWith("failed")) {
            if (intervalId) {
              clearInterval(intervalId);
              intervalId = null;
            }
          }
        } catch (statusError) {
          console.error("Error loading tree summary/status:", statusError);
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    loadTreeData();
    intervalId = setInterval(loadTreeData, 3000);

    return () => {
      isMounted = false;
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [documentId]);

  useEffect(() => {
    if (!queryId) {
      setRetrievalStats(null);
      return;
    }

    const loadRetrievalStats = async () => {
      try {
        const data = await getRetrievalSummary(queryId);
        setRetrievalStats(data);
      } catch (error) {
        console.error("Error loading retrieval summary:", error);
      }
    };

    loadRetrievalStats();
  }, [queryId]);

  return (
    <div className="h-full overflow-auto rounded-xl border border-[#B0BEC5]/30 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2 border-b border-[#B0BEC5]/20 pb-3">
        <FolderTree className="h-5 w-5 text-[#607D8B]" />
        <h2 className="font-medium text-[#333333]">Tree Structure</h2>
      </div>

      {isLoading ? (
        <div className="flex h-40 items-center justify-center text-sm text-[#B0BEC5]">
          Loading tree summary...
        </div>
      ) : !summary ? (
        <div className="flex h-40 items-center justify-center text-sm text-[#B0BEC5] text-center px-4">
          {buildStatus && buildStatus.toLowerCase().startsWith("failed")
            ? `Tree build failed: ${buildStatus}`
            : buildStatus && buildStatus !== "completed"
            ? `Document is ${buildStatus}. Building tree...`
            : "Select a document to see its hierarchical structure."}
        </div>
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-[#B0BEC5]/10 p-3 text-center">
              <p className="text-xs text-[#607D8B]">Total Nodes</p>
              <p className="text-xl font-bold text-[#333333]">{summary.total_nodes}</p>
            </div>
            <div className="rounded-lg bg-[#B0BEC5]/10 p-3 text-center">
              <p className="text-xs text-[#607D8B]">Max Depth</p>
              <p className="text-xl font-bold text-[#333333]">{summary.max_depth}</p>
            </div>
          </div>

          <div className="space-y-3">
            <h3 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[#B0BEC5]">
              <Layers className="h-3 w-3" />
              Level Distribution
            </h3>
            {summary.levels?.map((level) => {
              const numRetrieved = retrievalStats?.level_counts?.[level.level] || 0;
              const isRelevant = numRetrieved > 0;

              return (
                <div 
                  key={level.level}
                  className={`flex items-center gap-3 rounded-lg border p-3 transition-all ${
                    isRelevant 
                      ? "border-[#607D8B] bg-[#FFEBEE]/50 shadow-sm" 
                      : "border-[#B0BEC5]/20 bg-white"
                  }`}
                >
                  <div className={`flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold ${
                    isRelevant ? "bg-[#607D8B] text-white" : "bg-[#B0BEC5]/20 text-[#607D8B]"
                  }`}>
                    L{level.level}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-[#333333]">
                      {level.num_clusters} {level.num_clusters === 1 ? "Cluster" : "Clusters"}
                    </p>
                    {isRelevant && (
                      <p className="text-[10px] font-medium text-[#607D8B] flex items-center gap-1">
                        <Activity className="h-2.5 w-2.5" />
                        {numRetrieved} nodes retrieved here
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}