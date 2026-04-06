import { useEffect, useState } from "react";
import { X, FolderTree, Layers, Activity } from "lucide-react";
import { getTreeSummary, getDocumentStatus } from "../../lib/api";

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

interface TreeModalProps {
  documentId: string | null;
  onClose: () => void;
}

export function TreeModal({ documentId, onClose }: TreeModalProps) {
  const [summary, setSummary] = useState<TreeSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [buildStatus, setBuildStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!documentId) return;

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
          if (isMounted) {
            setBuildStatus(status.status);
          }
        } catch {
          console.error("Error loading tree status");
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

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="max-h-[90vh] w-full max-w-2xl overflow-auto rounded-lg bg-card">
        {/* Header */}
        <div className="sticky top-0 flex items-center justify-between border-b border-border bg-card px-6 py-4">
          <div className="flex items-center gap-3">
            <FolderTree className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Tree Structure</h2>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1 hover:bg-muted"
          >
            <X className="h-5 w-5 text-muted-foreground" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
              Loading tree summary...
            </div>
          ) : !summary ? (
            <div className="flex items-center justify-center py-12 text-center text-sm text-muted-foreground">
              {buildStatus && buildStatus.toLowerCase().startsWith("failed")
                ? `Tree build failed: ${buildStatus}`
                : buildStatus && buildStatus !== "completed"
                ? `Document is ${buildStatus}. Building tree...`
                : "Select a document to see its hierarchical structure."}
            </div>
          ) : (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg bg-primary/10 p-4 text-center">
                  <p className="text-xs text-muted-foreground">Total Nodes</p>
                  <p className="text-2xl font-bold text-foreground">{summary.total_nodes}</p>
                </div>
                <div className="rounded-lg bg-primary/10 p-4 text-center">
                  <p className="text-xs text-muted-foreground">Max Depth</p>
                  <p className="text-2xl font-bold text-foreground">{summary.max_depth}</p>
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  <Layers className="h-3 w-3" />
                  Level Distribution
                </h3>
                {summary.levels?.map((level) => (
                  <div 
                    key={level.level}
                    className="flex items-center gap-3 rounded-lg border border-border bg-background p-3 transition-all"
                  >
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20 text-xs font-bold text-primary">
                      L{level.level}
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-foreground">
                        {level.num_clusters} {level.num_clusters === 1 ? "Cluster" : "Clusters"}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
