import { Network } from "lucide-react";

export function Header() {
  return (
    <header className="border-b border-[#B0BEC5]/30 bg-white">
      <div className="mx-auto px-8 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#607D8B]">
            <Network className="h-4 w-4 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-medium tracking-tight text-[#333333]">
              EywaAI | TreeRAG
            </h1>
          </div>
        </div>
      </div>
    </header>
  );
}