import { useEffect, useState } from "react";
import { X, Layers, ChevronDown, ChevronUp } from "lucide-react";
import { getRetrievedNodes } from "../../lib/api";

interface RetrievedNode {
  node_id: string;
  level: number;
  text: string;
  similarity: number;
  title?: string | null;
  summary?: string | null;
}

interface NodesModalProps {
  queryId: string;
  onClose: () => void;
}

function RetrievedNodeCard({ node }: { node: RetrievedNode }) {
  const [expanded, setExpanded] = useState(false);

  const getLevelColor = (level: number) => {
    switch (level) {
      case 1:
        return "bg-primary/20 text-primary";
      case 2:
        return "bg-primary/15 text-primary";
      case 3:
        return "bg-primary/10 text-primary";
      default:
        return "bg-primary/5 text-primary";
    }
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.9) return "text-primary font-semibold";
    if (similarity >= 0.8) return "text-primary/80";
    return "text-muted-foreground";
  };

  return (
    <div className="rounded-lg border border-border bg-background p-4 shadow-sm transition-all hover:border-primary/30 hover:shadow-md">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="mb-2 flex items-center gap-2">
            <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${getLevelColor(node.level)}`}>
              Level {node.level}
            </span>
            <span className="text-xs text-muted-foreground">ID: {node.node_id}</span>
          </div>
          <p className="text-sm font-medium text-foreground">{node.title || "Context Node"}</p>
          <p className="mt-1 text-xs text-muted-foreground line-clamp-2">{node.summary || node.text.substring(0, 100) + "..."}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={`text-lg font-semibold ${getSimilarityColor(node.similarity)}`}>
            {(node.similarity * 100).toFixed(0)}%
          </span>
          <span className="text-xs text-muted-foreground">similarity</span>
        </div>
      </div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-1 text-xs text-primary transition-colors hover:text-primary/80"
      >
        {expanded ? (
          <>
            <ChevronUp className="h-3 w-3" />
            Hide full text
          </>
        ) : (
          <>
            <ChevronDown className="h-3 w-3" />
            Show full text
          </>
        )}
      </button>
      {expanded && (
        <div className="mt-3 rounded-md bg-primary/5 p-3 text-sm text-foreground whitespace-pre-wrap">
          {node.text}
        </div>
      )}
    </div>
  );
}

export function NodesModal({ queryId, onClose }: NodesModalProps) {
  const [nodes, setNodes] = useState<RetrievedNode[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!queryId) return;

    const loadNodes = async () => {
      setIsLoading(true);
      try {
        const data = await getRetrievedNodes(queryId);
        const allNodes = (data.levels || []).flatMap((l: any) => l.nodes);
        setNodes(allNodes);
      } catch (error) {
        console.error("Error loading retrieved nodes:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadNodes();
  }, [queryId]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="max-h-[90vh] w-full max-w-2xl overflow-auto rounded-lg bg-card">
        {/* Header */}
        <div className="sticky top-0 flex items-center justify-between border-b border-border bg-card px-6 py-4">
          <div className="flex items-center gap-3">
            <Layers className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Retrieved Nodes</h2>
            {!isLoading && (
              <span className="ml-auto rounded-full bg-primary/20 px-2.5 py-0.5 text-xs font-medium text-primary">
                {nodes.length} nodes
              </span>
            )}
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
              Retrieving nodes...
            </div>
          ) : nodes.length === 0 ? (
            <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
              No nodes retrieved.
            </div>
          ) : (
            <div className="space-y-3">
              {nodes.map((node, i) => (
                <RetrievedNodeCard key={`${node.node_id}-${i}`} node={node} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
