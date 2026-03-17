import os

from pipeline import RaptorPipeline

PDF_PATH = "./data/sample.pdf"
TREE_PATH = "./my_tree"


def _tree_exists(tree_path: str) -> bool:
    return (
        os.path.exists(os.path.join(tree_path, "tree.json"))
        and os.path.exists(os.path.join(tree_path, "embeddings.npz"))
    )


def main() -> None:
    pipe = RaptorPipeline(
        embed_model="multi-qa-mpnet-base-cos-v1",
        llm_model="llama3.2",
        strategy="collapsed",
        max_tokens=2000,
    )

    if _tree_exists(TREE_PATH):
        print(f"Loading existing tree from '{TREE_PATH}'...")
        pipe.load(TREE_PATH)
    else:
        print(f"Building tree from '{PDF_PATH}'.")
        pipe.build(pdf_path=PDF_PATH, save_path=TREE_PATH)

    print("Ready. Ask a question (type 'exit' to quit).")
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break
        if not query:
            continue

        context = pipe.retrieve(query)
        print("\nContext:\n")
        print(context)


if __name__ == "__main__":
    main()