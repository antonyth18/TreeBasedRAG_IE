import { ChevronDown, ChevronUp, Layers } from "lucide-react";
import { useEffect, useState } from "react";
import { getRetrievedNodes } from "../../lib/api";

interface RetrievedNode {
  node_id: string;
  level: number;
  text: string;
  similarity: number;
  title?: string | null;
  summary?: string | null;
}

interface RetrievalDetailsProps {
  queryId: string | null;
}

function RetrievedNodeCard({ node }: { node: RetrievedNode }) {
  const [expanded, setExpanded] = useState(false);

  const getLevelColor = (level: number) => {
    switch (level) {
      case 1:
        return "bg-[#FFEBEE] text-[#607D8B]";
      case 2:
        return "bg-[#B0BEC5]/20 text-[#607D8B]";
      case 3:
        return "bg-[#607D8B]/10 text-[#607D8B]";
      default:
        return "bg-[#B0BEC5]/10 text-[#607D8B]";
    }
  };

  const getSimilarityColor = (similarity: number) => {
    if (similarity >= 0.9) return "text-[#607D8B]";
    if (similarity >= 0.8) return "text-[#607D8B]/80";
    return "text-[#B0BEC5]";
  };

  return (
    <div className="rounded-lg border border-[#B0BEC5]/30 bg-white p-4 shadow-sm transition-all hover:border-[#607D8B]/30 hover:shadow-md">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="mb-2 flex items-center gap-2">
            <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${getLevelColor(node.level)}`}>
              Level {node.level}
            </span>
            <span className="text-xs text-[#B0BEC5]">ID: {node.node_id}</span>
          </div>
          <p className="text-sm font-medium text-[#333333]">{node.title || "Context Node"}</p>
          <p className="mt-1 text-xs text-[#607D8B] line-clamp-2">{node.summary || node.text.substring(0, 100) + "..."}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={`text-lg font-semibold ${getSimilarityColor(node.similarity)}`}>
            {(node.similarity * 100).toFixed(0)}%
          </span>
          <span className="text-xs text-[#B0BEC5]">similarity</span>
        </div>
      </div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-1 text-xs text-[#607D8B] transition-colors hover:text-[#607D8B]/80"
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
        <div className="mt-3 rounded-md bg-[#B0BEC5]/10 p-3 text-sm text-[#607D8B] whitespace-pre-wrap">
          {node.text}
        </div>
      )}
    </div>
  );
}

export function RetrievalDetails({ queryId }: RetrievalDetailsProps) {
  const [nodes, setNodes] = useState<RetrievedNode[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!queryId) {
      setNodes([]);
      return;
    }

    const loadNodes = async () => {
      setIsLoading(true);
      try {
        const data = await getRetrievedNodes(queryId);
        // data.levels is an array of {level, nodes}
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
    <div className="h-full overflow-auto rounded-xl border border-[#B0BEC5]/30 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2 border-b border-[#B0BEC5]/20 pb-3">
        <Layers className="h-5 w-5 text-[#607D8B]" />
        <h2 className="font-medium text-[#333333]">Retrieved Nodes</h2>
        {!isLoading && (
          <span className="ml-auto rounded-full bg-[#B0BEC5]/20 px-2.5 py-0.5 text-xs font-medium text-[#607D8B]">
            {nodes.length} nodes
          </span>
        )}
      </div>

      {isLoading ? (
        <div className="flex h-40 items-center justify-center text-sm text-[#B0BEC5]">
          Retrieving nodes...
        </div>
      ) : nodes.length === 0 ? (
        <div className="flex h-40 items-center justify-center text-sm text-[#B0BEC5]">
          {queryId ? "No nodes retrieved." : "Run a query to see details."}
        </div>
      ) : (
        <div className="space-y-3">
          {nodes.map((node, i) => (
            <RetrievedNodeCard key={`${node.node_id}-${i}`} node={node} />
          ))}
        </div>
      )}
    </div>
  );
}