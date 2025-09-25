import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import argparse

# Load environment variables
load_dotenv()

class QdrantRetriever:
    def __init__(self, collection_name="ufro_normativa"):
        """Initializes the Qdrant client and the embedding model."""
        self.qdrant_client = QdrantClient(
            url=os.environ.get("QDRANT_HOST"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = collection_name
        print("Retriever initialized. Connected to Qdrant.")

    def retrieve(self, query: str, k: int = 4):
        """
        Performs a semantic search on Qdrant and retrieves the most relevant chunks.
        Returns a list of dictionaries with text and metadata.
        """
        # 1. Convert the query into a vector (embedding)
        query_vector = self.embedding_model.encode(query).tolist()

        # 2. Perform the search in the Qdrant collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True
        )

        # 3. Format the results for use by the LLM
        retrieved_chunks = []
        for result in search_result:
            chunk_data = {
                "text": result.payload.get("text"),
                "score": result.score,
                "doc_id": result.payload.get("doc_id"),
                "title": result.payload.get("title"),
                "page": result.payload.get("page"),
                "url": result.payload.get("url"),
                "vigencia": result.payload.get("vigencia"),
            }
            retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks

def main():
    parser = argparse.ArgumentParser(description="Test chunk retrieval from Qdrant.")
    parser.add_argument("query", type=str, help="The query to search for in the database.")
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve.")
    args = parser.parse_args()
    
    retriever = QdrantRetriever()
    chunks = retriever.retrieve(args.query, args.k)
    
    if not chunks:
        print("No results found.")
        return
        
    print("\n--- Search Results ---")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] Document: {chunk['title']} (Page {chunk['page']})")
        print(f"URL: {chunk['url']}")
        print(f"Similarity Score: {chunk['score']:.4f}")
        # Print a snippet of the text
        if chunk['text']:
            print("Text Snippet:\n" + " ".join(chunk['text'].split()[:50]) + "...\n")
        else:
            print("Text: [Not available in payload]\n")

if __name__ == "__main__":
    main()