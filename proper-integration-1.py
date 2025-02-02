import subprocess
from dotenv import load_dotenv
import os
import re
import json
import pandas as pd
from typing import List, Dict, Union, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from phonetics import metaphone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

@dataclass
class MedicineMatch:
    name: str
    similarity: float
    match_type: str  # 'name', 'composition1', 'composition2'
    original_data: Dict
    phonetic_match: bool = False

class MedicineMatcher:
    def __init__(self, medicine_data: Union[str, pd.DataFrame]):
        """Initialize the matcher with medicine database"""
        if isinstance(medicine_data, str):
            self.df = pd.read_csv(medicine_data)
        else:
            self.df = medicine_data
            
        # Common medicine abbreviations mapping
        self.abbreviations = {
            'pcm': 'paracetamol',
            'acet': 'acetaminophen',
            'para': 'paracetamol',
            'cpm': 'chlorpheniramine',
            'asp': 'aspirin',
            'ibp': 'ibuprofen',
            'ctz': 'cetirizine',
            'dex': 'dexamethasone',
            'amx': 'amoxicillin'
        }
        
        # Build search index
        self.medicine_index = self._build_medicine_index()
    
    def _build_medicine_index(self) -> Dict:
        """Build search index from names and compositions"""
        index = {}
        
        for _, row in self.df.iterrows():
            data = row.to_dict()
            
            # Index main name
            name_terms = self._get_searchable_terms(row['name'])
            for term in name_terms:
                if term not in index:
                    index[term] = []
                index[term].append(('name', data))
            
            # Index composition1
            if pd.notna(row['short_composition1']):
                comp1_terms = self._get_searchable_terms(row['short_composition1'])
                for term in comp1_terms:
                    if term not in index:
                        index[term] = []
                    index[term].append(('composition1', data))
            
            # Index composition2
            if pd.notna(row['short_composition2']) and row['short_composition2']:
                comp2_terms = self._get_searchable_terms(row['short_composition2'])
                for term in comp2_terms:
                    if term not in index:
                        index[term] = []
                    index[term].append(('composition2', data))
        
        return index
    
    def _get_searchable_terms(self, text: str) -> Set[str]:
        """Extract searchable terms including abbreviations"""
        if pd.isna(text) or not text:
            return set()
        
        terms = set()
        processed = self._preprocess_name(text)
        terms.add(processed)
        terms.add(self._get_phonetic_code(processed))
        
        words = processed.split()
        for word in words:
            terms.add(word)
            if word in self.abbreviations:
                terms.add(self.abbreviations[word])
        
        return terms
    
    def _get_phonetic_code(self, name: str) -> str:
        """Get phonetic code for name"""
        return metaphone(name)
    
    def _preprocess_name(self, name: str) -> str:
        """Clean and standardize names"""
        if pd.isna(name):
            return ""
            
        name = str(name).lower()
        name = re.sub(r'\d+\s*(mg|ml|g|mcg|µg|kg|tablets|capsules)', '', name)
        name = re.sub(r'\([^)]*\)', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity"""
        return SequenceMatcher(None, str1, str2).ratio()
    
    @lru_cache(maxsize=1000)
    def find_matches(self, query_name: str, threshold: float = 0.7) -> List[MedicineMatch]:
        """Find matches using names and compositions"""
        processed_query = self._preprocess_name(query_name)
        query_terms = self._get_searchable_terms(query_name)
        matches = {}
        
        for term in query_terms:
            # Exact matches
            if term in self.medicine_index:
                for match_type, data in self.medicine_index[term]:
                    key = (data['name'], match_type)
                    matches[key] = MedicineMatch(
                        name=data['name'],
                        similarity=1.0,
                        match_type=match_type,
                        original_data=data
                    )
            
            # Similar matches
            for indexed_term, indexed_matches in self.medicine_index.items():
                similarity = self._calculate_similarity(term, indexed_term)
                if similarity >= threshold:
                    for match_type, data in indexed_matches:
                        key = (data['name'], match_type)
                        if key not in matches or matches[key].similarity < similarity:
                            matches[key] = MedicineMatch(
                                name=data['name'],
                                similarity=similarity,
                                match_type=match_type,
                                original_data=data
                            )
        
        result = list(matches.values())
        result.sort(key=lambda x: x.similarity, reverse=True)
        return result

class PrescriptionProcessor:
    def __init__(self, medicine_db_path: str):
        """Initialize prescription processor"""
        load_dotenv()
        self.api_key = os.getenv('TOGETHER_API_KEY')
        if self.api_key is None:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        
        self.medicine_matcher = MedicineMatcher(medicine_db_path)
        
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from prescription image"""
        result = subprocess.run(
            ['node', 'ocrScript.js', image_path, self.api_key],
            capture_output=True,
            text=True
        )
        return result.stdout
    
    def process_and_query(self, text: str) -> Dict:
        """Process prescription text and extract medications"""
        # Query to extract structured medication info
        query = """Extract only the medication-related information from the prescription text in a structured format. 
        For each medication, include:
        - Brand name (in parentheses)
        - Generic name
        - Strength
        - Dosage form
        - Quantity (number after #)
        - Instructions (after Sig:)

        Format as JSON with medications array."""
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=32,
            length_function=len,
        )
        texts = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        
        docs = docsearch.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)
        
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"medications": []}
    
    def process_prescription(self, image_path: str) -> Dict:
        """Process prescription image and find medicine matches"""
        # Extract text from image
        prescription_text = self.extract_text_from_image(image_path)
        
        # Extract structured medication data
        extracted_data = self.process_and_query(prescription_text)
        medications = extracted_data.get('medications', [])
        
        # Match each medication
        results = []
        for med in medications:
            matches = []
            
            # Try matching both brand and generic names
            if med.get('brand_name'):
                brand_matches = self.medicine_matcher.find_matches(med['brand_name'])
                matches.extend(brand_matches)
                
            if med.get('generic_name'):
                generic_matches = self.medicine_matcher.find_matches(med['generic_name'])
                matches.extend(generic_matches)
            
            # Deduplicate matches
            unique_matches = {m.name: m for m in matches}.values()
            
            results.append({
                "prescribed": {
                    "brand_name": med.get('brand_name', ''),
                    "generic_name": med.get('generic_name', ''),
                    "strength": med.get('strength', ''),
                    "dosage_form": med.get('dosage_form', ''),
                    "quantity": med.get('quantity', ''),
                    "instructions": med.get('instructions', '')
                },
                "matches": [
                    {
                        "name": match.name,
                        "similarity": match.similarity,
                        "match_type": match.match_type,
                        "details": {
                            "name": match.original_data['name'],
                            "price": match.original_data['price(₹)'],
                            "manufacturer": match.original_data['manufacturer_name'],
                            "composition1": match.original_data['short_composition1'],
                            "composition2": match.original_data['short_composition2'],
                            "is_discontinued": match.original_data['Is_discontinued'],
                            "pack_size": match.original_data['pack_size_label']
                        }
                    }
                    for match in unique_matches
                ]
            })
        
        return {
            "original_text": prescription_text,
            "results": results
        }

def main():
    # Initialize processor with medicine database
    processor = PrescriptionProcessor('medicines_P_200.csv')
    
    # Process prescription image
    image_path = r'C:\\Users\\ihamz\\curestock\\paracetamol.jpg'
    results = processor.process_prescription(image_path)
    
    # Print results
    print("\nPrescription Analysis Results:")
    print("=" * 50)
    print("\nExtracted Text:")
    print("-" * 30)
    print(results['original_text'])
    
    print("\nMatched Medicines:")
    print("-" * 30)
    for result in results['results']:
        print(f"\nPrescribed: {result['prescribed']['brand_name']} ({result['prescribed']['generic_name']})")
        print(f"Strength: {result['prescribed']['strength']}")
        print(f"Instructions: {result['prescribed']['instructions']}")
        print("\nMatches found:")
        for match in result['matches']:
            print(f"- {match['name']} (Similarity: {match['similarity']:.2f})")
            print(f"  Match type: {match['match_type']}")
            print(f"  Price: ₹{match['details']['price']}")
            print(f"  Composition: {match['details']['composition1']}, {match['details']['composition2']}")
            print(f"  Manufacturer: {match['details']['manufacturer']}")
            print(f"  Pack size: {match['details']['pack_size']}")
            print()

if __name__ == "__main__":
    main()