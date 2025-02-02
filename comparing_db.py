import pandas as pd
import re
from typing import List, Dict, Union
from difflib import SequenceMatcher
from dataclasses import dataclass

@dataclass
class MedicineMatch:
    name: str
    similarity: float
    original_data: Dict

class MedicineMatcher:
    def __init__(self, medicine_data: Union[str, pd.DataFrame]):
        """
        Initialize the matcher with medicine data
        Args:
            medicine_data: Either a DataFrame or path to CSV file
        """
        if isinstance(medicine_data, str):
            self.df = pd.read_csv(medicine_data)
        else:
            self.df = medicine_data
            
        # Create processed names dictionary for quick lookup
        self.processed_names = {
            self._preprocess_name(row['name']): row.to_dict()
            for _, row in self.df.iterrows()
        }
        
    def _preprocess_name(self, name: str) -> str:
        """
        Clean and standardize medicine names
        """
        # Convert to lowercase
        name = name.lower()
        
        # Remove dosage information (e.g., 500mg, 10ml)
        name = re.sub(r'\d+\s*(mg|ml|g|mcg|µg|kg|tablets|capsules)', '', name)
        
        # Remove parenthetical information
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Remove extra whitespace and standardize spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using SequenceMatcher
        """
        return SequenceMatcher(None, str1, str2).ratio()
    
    def find_matches(self, query_name: str, threshold: float = 0.7) -> List[MedicineMatch]:
        """
        Find matches for a given medicine name
        Args:
            query_name: Name to search for
            threshold: Minimum similarity score (0-1)
        Returns:
            List of MedicineMatch objects sorted by similarity
        """
        processed_query = self._preprocess_name(query_name)
        matches = []
        
        # First try exact matching with processed names
        for proc_name, original_data in self.processed_names.items():
            if processed_query in proc_name or proc_name in processed_query:
                matches.append(MedicineMatch(
                    name=original_data['name'],
                    similarity=1.0,
                    original_data=original_data
                ))
                continue
                
            # If no exact match, calculate similarity
            similarity = self._calculate_similarity(processed_query, proc_name)
            if similarity >= threshold:
                matches.append(MedicineMatch(
                    name=original_data['name'],
                    similarity=similarity,
                    original_data=original_data
                ))
        
        # Sort by similarity score in descending order
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches

def example_usage():
    # Sample data
    data = {
        'id': [1, 2, 3],
        'name': ['Paracetamol 500mg', 'Amoxicillin 250mg', 'Ibuprofen 400mg'],
        'price(₹)': [10.5, 15.75, 12.0],
        'Is_discontinued': [False, False, False],
        'manufacturer_name': ['Mfg1', 'Mfg2', 'Mfg3'],
        'type': ['tablet', 'capsule', 'tablet'],
        'pack_size_label': ['10s', '6s', '15s'],
        'short_composition1': ['comp1', 'comp2', 'comp3'],
        'short_composition2': ['', '', '']
    }
    df = pd.DataFrame(data)
    
    # Initialize matcher
    matcher = MedicineMatcher(df)
    
    # Example searches
    test_queries = [
        "Paracetamol",
        "paracetamol 500",
        "Amox",
        "ibuprofen tablet"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: {query}")
        matches = matcher.find_matches(query)
        for match in matches:
            print(f"Match: {match.name} (Similarity: {match.similarity:.2f})")

if __name__ == "__main__":
    example_usage()