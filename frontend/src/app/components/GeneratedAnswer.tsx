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
    <div className="h-full overflow-auto rounded-xl border border-border bg-card p-5 shadow-sm">
      <div className="mb-4 flex items-center gap-2 border-b border-border pb-3">
        <MessageSquare className="h-5 w-5 text-primary" />
        <h2 className="font-medium text-foreground">Answer</h2>
      </div>
      
      <div className="mb-6 rounded-lg bg-primary/10 p-5 min-h-[100px] flex items-center justify-center">
        {isLoading ? (
          <div className="text-primary animate-pulse">Generating answer...</div>
        ) : answer ? (
          <p className="whitespace-pre-line leading-relaxed text-foreground w-full">
            {answer}
          </p>
        ) : (
          <p className="text-muted-foreground italic text-sm">
            Ask a question to see the generated answer.
          </p>
        )}
      </div>

      {sources.length > 0 && (
        <div className="border-t border-border pt-5">
          <h3 className="mb-3 text-sm font-medium text-foreground">Sources</h3>
          <div className="space-y-2">
            {sources.map((source, i) => (
              <div
                key={`${source.node_id}-${i}`}
                className="flex items-start gap-3 rounded-lg border border-border bg-primary/5 p-3 transition-all hover:border-primary/30 hover:bg-primary/10"
              >
                <div className="flex-1">
                  <div className="mb-1 flex items-center gap-2">
                    <span className="text-xs font-medium text-foreground">
                      {source.node_id}
                    </span>
                    <span className="text-xs text-muted-foreground">• Level {source.level}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">{source.snippet}</p>
                </div>
                <button className="text-muted-foreground transition-colors hover:text-primary">
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