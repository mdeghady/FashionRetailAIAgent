import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Any
import re
import uuid


def normalize_sku(sku):
    # Function to normalize SKUs by removing non-alphanumeric characters
    if isinstance(sku, str):
        return re.sub(r'\W+', '', sku)  # \W matches any non-word character
    return None


class FashionRAGSystem:
    def __init__(self, embedding_model="multi-qa-MiniLM-L6-cos-v1"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.PersistentClient(path='./chroma_db')
        self.collection = None

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the fashion dataset"""

        # Create a copy to avoid modifying original
        processed_df = df.copy()

        # Clean and standardize text fields
        processed_df['Brand'] = processed_df['Brand'].str.strip().str.title()
        processed_df['ProductName'] = processed_df['ProductName'].str.strip()

        # Handle missing descriptions
        processed_df['Description'] = processed_df['Description'].fillna('No description available')

        # Create price tier categories
        processed_df['price_tier'] = processed_df['CurrentPrice'].apply(self._categorize_price)

        # Calculate discount percentage
        processed_df['discount_percentage'] = (
                (processed_df['OriginalPrice'] - processed_df['CurrentPrice']) /
                processed_df['OriginalPrice'] * 100
        ).round(2)

        # Clean and standardize availability
        processed_df['StockAvailability'] = processed_df['StockAvailability'].fillna('Unknown')

        # Normalize SKU
        processed_df['NormalizedSKU'] = processed_df['sku'].apply(normalize_sku)

        return processed_df

    def _categorize_price(self, price: float) -> str:
        """Categorize products into price tiers"""
        if pd.isna(price):
            return "unknown"
        elif price < 50:
            return "budget"
        elif price < 200:
            return "mid-range"
        else:
            return "luxury"

    def create_product_documents(self, df: pd.DataFrame) -> List[str]:
        """Transform tabular data into text documents for embedding"""

        documents = []
        for _, row in df.iterrows():
            doc_parts = []

            # Core product information
            doc_parts.append(f"{row['Brand']} {row['ProductName']}")

            # Add description if available
            if pd.notna(row['Description']) and row['Description'] != 'No description available':
                doc_parts.append(f"Description: {row['Description']}")

            # Add categorical context
            doc_parts.append(f"Category: {row['Category']} in {row['Department']} department")

            # Add color information
            if pd.notna(row['ProductColor']):
                doc_parts.append(f"Color: {row['ProductColor']}")

            # Add size availability
            if pd.notna(row['AvailableSizes']):
                doc_parts.append(f"Available sizes: {row['AvailableSizes']}")

            # Add price context
            doc_parts.append(f"Price tier: {row['price_tier']}")

            # Add stock status
            doc_parts.append(f"Stock: {row['StockAvailability']}")

            # Join all parts
            document = ". ".join(doc_parts)
            documents.append(document)

        return documents

    def create_metadata(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create metadata dictionaries for each product"""

        metadata_list = []
        for _, row in df.iterrows():
            metadata = {
                'normalized_sku': str(row['NormalizedSKU']),
                'brand': str(row['Brand']),
                'product_name': str(row['ProductName']),
                'category': str(row['Category']),
                'department': str(row['Department']),
                'current_price': float(row['CurrentPrice']) if pd.notna(row['CurrentPrice']) else 0.0,
                'original_price': float(row['OriginalPrice']) if pd.notna(row['OriginalPrice']) else 0.0,
                'currency': str(row['PriceCurrency']) if pd.notna(row['PriceCurrency']) else 'USD',
                'color': str(row['ProductColor']) if pd.notna(row['ProductColor']) else '',
                'sizes': str(row['AvailableSizes']) if pd.notna(row['AvailableSizes']) else '',
                'stock_status': str(row['StockAvailability']),
                'website': str(row['website']),
                'date': str(row['date']),
                'price_tier': row['price_tier'],
                'discount_percentage': float(row['discount_percentage']) if pd.notna(
                    row['discount_percentage']) else 0.0,
                'product_url': str(row['ProductURL']) if pd.notna(row['ProductURL']) else ''
            }
            metadata_list.append(metadata)

        return metadata_list

    def generate_embeddings(self, documents: List[str]) -> np.ndarray:
        """Generate vector embeddings for product documents"""

        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(
            documents,
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def setup_vector_database(self, collection_name: str = "fashion_products"):
        """Initialize the vector database collection"""

        try:
            # Delete collection if it exists
            self.chroma_client.delete_collection(name=collection_name)
        except:
            pass

        # Create new collection
        self.collection = self.chroma_client.create_collection(name=collection_name)
        print(f"Created collection: {collection_name}")

    def insert_data(self, embeddings: np.ndarray, documents: List[str],
                    metadata_list: List[Dict[str, Any]]):
        """Insert embeddings and metadata into vector database"""

        # Create unique IDs for each product
        ids = [f"{meta['normalized_sku']}_{meta['date']}_{meta['website']}_{uuid.uuid4()}" for meta in metadata_list]

        # Insert data in batches
        batch_size = 1000 if len(embeddings) > 5000 else 500
        total_batches = len(embeddings) // batch_size + (1 if len(embeddings) % batch_size > 0 else 0)

        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_num = i // batch_size + 1

            print(f"Inserting batch {batch_num}/{total_batches}")

            self.collection.add(
                embeddings=embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadata_list[i:batch_end],
                ids=ids[i:batch_end]
            )

        print(f"Successfully inserted {len(embeddings)} products into vector database")

    def similarity_search(self, query: str, n_results: int = 10,
                          filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform semantic search on the fashion products"""

        if self.collection is None:
            raise ValueError("Vector database not initialized. Call setup_vector_database() first.")

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])

        # Prepare filters for ChromaDB format
        where_filter = None
        if filters:
            where_filter = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    where_filter[key] = {"$in": value}
                else:
                    where_filter[key] = value

        # Adding and operator if there are multiple filters
        if where_filter and len(where_filter) > 1:
            where_filter = {"$and": [{k: v} for k, v in where_filter.items()]}

        # Perform search
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def process_and_insert_dataset(self, df: pd.DataFrame, collection_name: str = "fashion_products"):
        """Complete pipeline to process and insert fashion dataset"""

        print("Step 1: Preprocessing data...")
        processed_df = self.preprocess_data(df)

        print("Step 2: Creating product documents...")
        documents = self.create_product_documents(processed_df)

        print("Step 3: Creating metadata...")
        metadata_list = self.create_metadata(processed_df)

        print("Step 4: Generating embeddings...")
        embeddings = self.generate_embeddings(documents)

        print("Step 5: Setting up vector database...")
        self.setup_vector_database(collection_name)

        print("Step 6: Inserting data...")
        self.insert_data(embeddings, documents, metadata_list)

        print("✅ Fashion RAG system setup complete!")
        return len(embeddings)


# Example usage and search demonstration
def demonstrate_search(rag_system: FashionRAGSystem):
    """Demonstrate different types of searches"""

    print("\n" + "=" * 50)
    print("SEARCH DEMONSTRATIONS")
    print("=" * 50)

    # 1. Basic semantic search
    print("\n1. Semantic search for 'comfortable running shoes':")
    results = rag_system.similarity_search("comfortable running shoes", n_results=5)

    for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
    )):
        print(f"\n  Result {i + 1} (similarity: {1 - distance:.3f}):")
        print(f"  Product: {metadata['brand']} {metadata['product_name']}")
        print(f"  Price: {metadata['currency']} {metadata['current_price']}")
        print(f"  Category: {metadata['category']}")
        print(f"  Description: {doc[:100]}...")

    # 2. Filtered search
    print("\n\n2. Search for 'stylish jacket' under $100:")
    results = rag_system.similarity_search(
        "stylish jacket",
        n_results=3,
        filters={
            "current_price": {"$lt": 100},
            "category": "Jackets"
        }
    )

    for i, (metadata, distance) in enumerate(zip(
            results['metadatas'][0],
            results['distances'][0]
    )):
        print(f"\n  Result {i + 1}:")
        print(f"  Product: {metadata['brand']} {metadata['product_name']}")
        print(f"  Price: {metadata['currency']} {metadata['current_price']}")
        print(f"  Discount: {metadata['discount_percentage']:.1f}%")


# Example of how to use the system
if __name__ == "__main__":
    # Initialize the RAG system
    fashion_rag = FashionRAGSystem()

    # Load your dataset
    df = pd.read_parquet('all_data.parquet')

    print("Loaded dataset with shape:", df.shape)

    # For demonstration, create a sample dataset
    # sample_data = {
    #     'Brand': ['Nike', 'Adidas', 'Zara', 'H&M', 'Nike'],
    #     'ProductName': ['Air Max 270', 'Ultraboost 22', 'Wool Coat', 'Cotton T-Shirt', 'React Infinity'],
    #     'ProductColor': ['Black', 'White', 'Navy', 'Red', 'Blue'],
    #     'CurrentPrice': [130, 180, 199, 15, 160],
    #     'OriginalPrice': [150, 200, 249, 20, 160],
    #     'Department': ['Shoes', 'Shoes', 'Outerwear', 'Tops', 'Shoes'],
    #     'Category': ['Sneakers', 'Running', 'Coats', 'T-Shirts', 'Running'],
    #     'Description': [
    #         'Comfortable athletic shoe with air cushioning',
    #         'High-performance running shoe with boost technology',
    #         'Elegant wool coat for winter season',
    #         'Basic cotton t-shirt for everyday wear',
    #         'Premium running shoe with react foam'
    #     ],
    #     'AvailableSizes': ['7,8,9,10,11', '8,9,10,11,12', 'S,M,L,XL', 'XS,S,M,L,XL', '7,8,9,10,11'],
    #     'StockAvailability': ['In Stock', 'Low Stock', 'In Stock', 'In Stock', 'Out of Stock'],
    #     'NormalizedSKU': ['NIKE_AM270_001', 'ADIDAS_UB22_001', 'ZARA_WC_001', 'HM_CT_001', 'NIKE_RI_001'],
    #     'PriceCurrency': ['USD', 'USD', 'USD', 'USD', 'USD'],
    #     'ProductURL': ['nike.com/air-max-270', 'adidas.com/ultraboost', 'zara.com/wool-coat', 'hm.com/t-shirt', 'nike.com/react'],
    #     'website': ['Nike', 'Adidas', 'Zara', 'H&M', 'Nike'],
    #     'date': ['2024-01-15', '2024-01-15', '2024-01-15', '2024-01-15', '2024-01-15'],
    #     'ProductCode': ['AM270', 'UB22', 'WC001', 'CT001', 'RI001'],
    #     'ProductColorCode': ['BLK', 'WHT', 'NVY', 'RED', 'BLU'],
    #     'WebCode': ['NKE001', 'ADS001', 'ZAR001', 'HM001', 'NKE002'],
    #     'sku': ['NIKE-AM270-BLK', 'ADIDAS-UB22-WHT', 'ZARA-WC-NVY', 'HM-CT-RED', 'NIKE-RI-BLU'],
    #     'Unnamed: 0': [0, 1, 2, 3, 4]
    # }
    #
    # df = pd.DataFrame(sample_data)

    # Process and insert the dataset
    num_products = fashion_rag.process_and_insert_dataset(df)
    print(f"\nProcessed {num_products} products successfully!")

    # Demonstrate search capabilities
    demonstrate_search(fashion_rag)