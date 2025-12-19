"""
Tourism Vector Database Query Interface
Advanced querying capabilities for the ShivYatra ChromaDB vector store
"""

import chromadb
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Add config to path
sys.path.append(str(Path(__file__).parent.parent / "config"))
from db_config import *

class TourismQueryEngine:
    """
    Advanced query engine for tourism vector database
    Provides semantic search, filtering, and recommendation capabilities
    """
    
    def __init__(self):
        """Initialize query engine"""
        self.client = None
        self.collection = None
        self.is_connected = False
    
    def connect(self) -> bool:
        """Connect to existing ChromaDB"""
        try:
            print("Connecting to ChromaDB...")
            
            self.client = chromadb.PersistentClient(
                path=str(CHROMA_DB_PATH),
                settings=chromadb.Settings(**CHROMA_SETTINGS)
            )
            
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
            self.is_connected = True
            
            count = self.collection.count()
            print(f"Connected to ChromaDB!")
            print(f"Collection: {COLLECTION_NAME}")
            print(f"Total entries: {count:,}")
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to ChromaDB: {str(e)}")
            return False
    
    def semantic_search(self, query: str, limit: int = 10, 
                       include_similarity: bool = True) -> List[Dict[str, Any]]:
        """Perform semantic search using natural language query"""
        if not self.is_connected:
            print("Database not connected!")
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'rank': i + 1
                    }
                    
                    if include_similarity and 'distances' in results:
                        # Convert distance to similarity score (ChromaDB uses cosine distance)
                        distance = results['distances'][0][i]
                        similarity = 1 - distance
                        result['similarity'] = round(similarity, 4)
                    
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return []
    
    def filter_search(self, query: str, filters: Dict[str, Any], 
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search with metadata filtering
        
        Args:
            query: Search query
            filters: Metadata filters (e.g., {"state": "Himachal Pradesh"})
            limit: Maximum results
        
        Returns:
            Filtered search results
        """
        if not self.is_connected:
            print("Database not connected!")
            return []
        
        try:
            # Build ChromaDB where clause
            where_clause = {}
            for key, value in filters.items():
                if isinstance(value, str):
                    where_clause[key] = {"$eq": value}
                elif isinstance(value, list):
                    where_clause[key] = {"$in": value}
                elif isinstance(value, dict):
                    where_clause[key] = value
            
            results = self.collection.query(
                query_texts=[query],
                where=where_clause,
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'rank': i + 1,
                        'similarity': round(1 - results['distances'][0][i], 4)
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Filtered search failed: {str(e)}")
            return []
    
    def get_recommendations(self, preferences: Dict[str, Any], 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get tourism recommendations based on preferences
        
        Args:
            preferences: User preferences
                - traveler_type: "solo", "family", "adventure"
                - budget: "low", "medium", "high"
                - interests: List of interest keywords
                - states: List of preferred states
        
        Returns:
            Personalized recommendations
        """
        if not self.is_connected:
            return []
        
        try:
            # Build query based on preferences
            query_parts = []
            filters = {}
            
            # Add traveler type preferences
            if preferences.get('traveler_type'):
                query_parts.append(f"{preferences['traveler_type']} traveler activities")
            
            # Add interest keywords
            if preferences.get('interests'):
                query_parts.extend(preferences['interests'])
            
            # Add budget filter
            if preferences.get('budget'):
                filters['price_range'] = preferences['budget']
            
            # Add state filter
            if preferences.get('states'):
                filters['state'] = preferences['states']
            
            # Construct query
            query = " ".join(query_parts) if query_parts else "tourism activities"
            
            if filters:
                return self.filter_search(query, filters, limit)
            else:
                return self.semantic_search(query, limit)
                
        except Exception as e:
            print(f"Recommendation failed: {str(e)}")
            return []
    
    def get_location_insights(self, location: str) -> Dict[str, Any]:
        """Get comprehensive insights about a specific location"""
        try:
            # Search for location-specific content
            results = self.filter_search(
                query=f"activities places {location}",
                filters={"city": location},
                limit=50
            )
            
            # If no results for city, try state
            if not results:
                results = self.filter_search(
                    query=f"activities places {location}",
                    filters={"state": location},
                    limit=50
                )
            
            if not results:
                return {"location": location, "insights": "No data found"}
            
            # Analyze results
            categories = [r['metadata']['category'] for r in results]
            subcategories = [r['metadata']['subcategory'] for r in results]
            price_ranges = [r['metadata']['price_range'] for r in results]
            
            # Calculate averages
            adventure_scores = [r['metadata']['adventure_score'] for r in results]
            family_scores = [r['metadata']['family_score'] for r in results]
            solo_scores = [r['metadata']['solo_traveler_score'] for r in results]
            
            insights = {
                "location": location,
                "total_activities": len(results),
                "top_categories": list(set(categories))[:5],
                "popular_subcategories": list(set(subcategories))[:5],
                "budget_distribution": {
                    "budget": price_ranges.count("budget"),
                    "unknown": price_ranges.count("unknown"), 
                    "mid_range": price_ranges.count("mid_range")
                },
                "traveler_suitability": {
                    "adventure": round(sum(adventure_scores) / len(adventure_scores), 2),
                    "family": round(sum(family_scores) / len(family_scores), 2),
                    "solo": round(sum(solo_scores) / len(solo_scores), 2)
                },
                "sample_activities": [r['content'][:100] + "..." for r in results[:3]]
            }
            
            return insights
            
        except Exception as e:
            print(f"Location insights failed: {str(e)}")
            return {"error": str(e)}


def interactive_query_demo():
    """Interactive demo of query capabilities"""
    print("TOURISM VECTOR DATABASE - INTERACTIVE DEMO")
    print("=" * 50)
    
    query_engine = TourismQueryEngine()
    
    if not query_engine.connect():
        print("Failed to connect to database!")
        return
    
    # Demo queries
    demo_queries = [
        {
            "type": "Semantic Search",
            "query": "adventure activities in mountains",
            "description": "Find mountain adventure activities"
        },
        {
            "type": "Filtered Search", 
            "query": "family friendly places",
            "filters": {"state": "Himachal"},
            "description": "Family activities in Himachal Pradesh"
        },
        {
            "type": "Recommendations",
            "preferences": {
                "traveler_type": "solo",
                "interests": ["trekking", "temples", "nature"],
                "budget": "budget"
            },
            "description": "Solo traveler recommendations"
        },
        {
            "type": "Location Insights",
            "location": "Manali",
            "description": "Insights about Manali"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n--- Demo {i}: {demo['type']} ---")
        print(f"Description: {demo['description']}")
        
        if demo['type'] == "Semantic Search":
            results = query_engine.semantic_search(demo['query'], limit=3)
            print_search_results(results)
            
        elif demo['type'] == "Filtered Search":
            results = query_engine.filter_search(demo['query'], demo['filters'], limit=3)
            print_search_results(results)
            
        elif demo['type'] == "Recommendations":
            results = query_engine.get_recommendations(demo['preferences'], limit=3)
            print_search_results(results)
            
        elif demo['type'] == "Location Insights":
            insights = query_engine.get_location_insights(demo['location'])
            print_location_insights(insights)
        
        print("─" * 50)


def print_search_results(results: List[Dict[str, Any]]):
    """Print formatted search results"""
    if not results:
        print("No results found")
        return
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"\n  Rank {result['rank']} | Similarity: {result.get('similarity', 'N/A')}")
        print(f"     Location: {result['metadata']['city']}, {result['metadata']['state']}")
        print(f"     Category: {result['metadata']['category']} → {result['metadata']['subcategory']}")
        print(f"     Budget: {result['metadata']['price_range']}")
        print(f"     Content: {result['content'][:120]}...")


def print_location_insights(insights: Dict[str, Any]):
    """Print formatted location insights"""
    if "error" in insights:
        print(f"Error: {insights['error']}")
        return
    
    print(f"Location: {insights['location']}")
    print(f"Activities: {insights['total_activities']}")
    print(f"Categories: {', '.join(insights['top_categories'])}")
    print(f"Budget: Budget({insights['budget_distribution']['budget']}) | Unknown({insights['budget_distribution']['unknown']}) | Mid-range({insights['budget_distribution']['mid_range']})")
    print(f"Best for: Adventure({insights['traveler_suitability']['adventure']}) | Family({insights['traveler_suitability']['family']}) | Solo({insights['traveler_suitability']['solo']})") 


if __name__ == "__main__":
    interactive_query_demo()