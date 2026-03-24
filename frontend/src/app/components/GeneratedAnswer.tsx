import { MessageSquare, ExternalLink } from "lucide-react";
import { useEffect, useState } from "react";
import { getRetrievedNodes } from "../../lib/api";

interface Source {
  node_id: string;
  level: number;
  snippet: string;
}

interface GeneratedAnswerProps {
  answer: string;
  queryId?: string | null;
  isLoading?: boolean;
}

export function GeneratedAnswer({ answer, queryId, isLoading }: GeneratedAnswerProps) {
  const [sources, setSources] = useState<Source[]>([]);

  useEffect(() => {
    if (!queryId) {
      setSources([]);
      return;
    }

    const loadSources = async () => {
      try {
        const data = await getRetrievedNodes(queryId);
        // Take top nodes as sources
        const allNodes = (data.levels || []).flatMap((l: any) => l.nodes);
        setSources(allNodes.slice(0, 3).map((n: any) => ({
          node_id: n.node_id,
          level: n.level,
          snippet: n.summary || n.text.substring(0, 100) + "..."
        })));
      } catch (error) {
        console.error("Error loading sources:", error);
      }
    };

    loadSources();
  }, [queryId]);

  return (
    <div className="h-full overflow-auto rounded-xl border border-[#B0BEC5]/30 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2 border-b border-[#B0BEC5]/20 pb-3">
        <MessageSquare className="h-5 w-5 text-[#607D8B]" />
        <h2 className="font-medium text-[#333333]">Answer</h2>
      </div>
      
      <div className="mb-6 rounded-lg bg-[#FFEBEE]/50 p-5 min-h-[100px] flex items-center justify-center">
        {isLoading ? (
          <div className="text-[#607D8B] animate-pulse">Generating answer...</div>
        ) : answer ? (
          <p className="whitespace-pre-line leading-relaxed text-[#333333] w-full">
            {answer}
          </p>
        ) : (
          <p className="text-[#B0BEC5] italic text-sm">
            Ask a question to see the generated answer.
          </p>
        )}
      </div>

      {sources.length > 0 && (
        <div className="border-t border-[#B0BEC5]/20 pt-5">
          <h3 className="mb-3 text-sm font-medium text-[#333333]">Sources</h3>
          <div className="space-y-2">
            {sources.map((source, i) => (
              <div
                key={`${source.node_id}-${i}`}
                className="flex items-start gap-3 rounded-lg border border-[#B0BEC5]/30 bg-[#B0BEC5]/5 p-3 transition-all hover:border-[#607D8B]/30 hover:bg-[#B0BEC5]/10"
              >
                <div className="flex-1">
                  <div className="mb-1 flex items-center gap-2">
                    <span className="text-xs font-medium text-[#333333]">
                      {source.node_id}
                    </span>
                    <span className="text-xs text-[#B0BEC5]">• Level {source.level}</span>
                  </div>
                  <p className="text-xs text-[#607D8B]">{source.snippet}</p>
                </div>
                <button className="text-[#B0BEC5] transition-colors hover:text-[#607D8B]">
                  <ExternalLink className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}