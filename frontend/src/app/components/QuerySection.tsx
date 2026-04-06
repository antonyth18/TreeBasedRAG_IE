import { Send } from "lucide-react";

interface QuerySectionProps {
  query: string;
  onQueryChange: (query: string) => void;
  onRunQuery: () => void;
  isDisabled?: boolean;
}

export function QuerySection({ query, onQueryChange, onRunQuery, isDisabled }: QuerySectionProps) {
  return (
    <>
      <div className={`flex items-center gap-2 rounded-full border border-border bg-muted px-4 py-3 shadow-sm ${isDisabled ? 'opacity-50' : ''}`}>
          <input
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Ask a question about your document..."
            disabled={isDisabled}
            className="flex-1 bg-transparent py-2 text-foreground placeholder-muted-foreground focus:outline-none disabled:cursor-not-allowed text-sm"
            onKeyDown={(e) => e.key === 'Enter' && !isDisabled && onRunQuery()}
          />
          <button
            onClick={onRunQuery}
            disabled={isDisabled || !query.trim()}
            className="flex items-center justify-center rounded-full bg-primary px-3 py-2 transition-all hover:bg-primary/90 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Send className="h-4 w-4 text-primary-foreground" />
          </button>
        </div>
    </>
  );
}