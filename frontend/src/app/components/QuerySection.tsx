import { Send } from "lucide-react";

interface QuerySectionProps {
  query: string;
  onQueryChange: (query: string) => void;
  onRunQuery: () => void;
  isDisabled?: boolean;
}

export function QuerySection({ query, onQueryChange, onRunQuery, isDisabled }: QuerySectionProps) {
  return (
    <div className="border-b border-[#B0BEC5]/30 bg-white">
      <div className="mx-auto max-w-3xl px-6 py-6">
        <div className={`flex items-center gap-2 rounded-full border border-[#B0BEC5] bg-white px-4 py-2 shadow-sm ${isDisabled ? 'opacity-50' : ''}`}>
          <input
            type="text"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Ask a question about your document..."
            disabled={isDisabled}
            className="flex-1 bg-transparent py-1 text-[#333333] placeholder-[#B0BEC5] focus:outline-none disabled:cursor-not-allowed"
            onKeyDown={(e) => e.key === 'Enter' && !isDisabled && onRunQuery()}
          />
          <button
            onClick={onRunQuery}
            disabled={isDisabled || !query.trim()}
            className="flex items-center justify-center rounded-full bg-[#607D8B] p-2 transition-all hover:bg-[#607D8B]/80 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Send className="h-4 w-4 text-white" />
          </button>
        </div>
      </div>
    </div>
  );
}