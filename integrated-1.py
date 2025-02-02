import subprocess
from dotenv import load_dotenv
import os
import re
import json
import pandas as pd
from typing import List, Dict, Union
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
    original_data: Dict
    phonetic_match: bool = False

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
        self.processed_names = {}
        self.phonetic_codes = {}
        
        for _, row in self.df.iterrows():
            processed_name = self._preprocess_name(row['name'])
            phonetic_code = self._get_phonetic_code(processed_name)
            
            self.processed_names[processed_name] = row.to_dict()
            self.phonetic_codes[phonetic_code] = row.to_dict()
    
    def _get_phonetic_code(self, name: str) -> str:
        """
        Get phonetic code for name using Metaphone algorithm
        """
        return metaphone(name)
    
    def _preprocess_name(self, name: str) -> str:
        """
        Clean and standardize medicine names
        """
        # Convert to lowercase
        name = name.lower()
        
        # Remove dosage information
        name = re.sub(r'\d+\s*(mg|ml|g|mcg|µg|kg|tablets|capsules)', '', name)
        
        # Remove parenthetical information
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using SequenceMatcher
        """
        return SequenceMatcher(None, str1, str2).ratio()
    
    @lru_cache(maxsize=1000)
    def find_matches(self, query_name: str, threshold: float = 0.7) -> List[MedicineMatch]:
        """
        Find matches for a given medicine name using both string similarity and phonetic matching
        """
        processed_query = self._preprocess_name(query_name)
        query_phonetic = self._get_phonetic_code(processed_query)
        matches = []
        
        # Try exact matching
        for proc_name, original_data in self.processed_names.items():
            if processed_query in proc_name or proc_name in processed_query:
                matches.append(MedicineMatch(
                    name=original_data['name'],
                    similarity=1.0,
                    original_data=original_data
                ))
                continue
            
            # Try phonetic matching
            if query_phonetic == self._get_phonetic_code(proc_name):
                matches.append(MedicineMatch(
                    name=original_data['name'],
                    similarity=0.9,  # High similarity for phonetic matches
                    original_data=original_data,
                    phonetic_match=True
                ))
                continue
            
            # Calculate string similarity
            similarity = self._calculate_similarity(processed_query, proc_name)
            if similarity >= threshold:
                matches.append(MedicineMatch(
                    name=original_data['name'],
                    similarity=similarity,
                    original_data=original_data
                ))
        
        # Sort by similarity score
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches

class PrescriptionProcessor:
    def __init__(self, medicine_db_path: str):
        """
        Initialize the prescription processor
        Args:
            medicine_db_path: Path to medicine database CSV
        """
        load_dotenv()
        self.api_key = os.getenv('TOGETHER_API_KEY')
        if self.api_key is None:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        
        self.medicine_matcher = MedicineMatcher(medicine_db_path)
        
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from prescription image using OCR
        """
        result = subprocess.run(
            ['node', 'ocrScript.js', image_path, self.api_key],
            capture_output=True,
            text=True
        )
        return result.stdout
    
    def process_and_query(self, text: str, query: str) -> str:
        """
        Process extracted text and run query
        """
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
        return chain.run(input_documents=docs, question=query)
    
    def match_medicines_from_prescription(self, prescription_text: str) -> Dict:
        """
        Extract medicines from prescription and match with database
        """
        # Query to extract medication information
        query = """Extract only the medication-related information from the prescription text in a structured format. 
        For each medication, include: Brand name, Generic name, Strength, Dosage form, Quantity, Instructions.
        Format as JSON with medications array."""
        
        # Extract structured medication data
        extracted_json = self.process_and_query(prescription_text, query)
        
        try:
            medications = json.loads(extracted_json)['medications']
        except (json.JSONDecodeError, KeyError):
            return {"error": "Failed to parse medication information"}
        
        # Match each medication with database
        results = []
        for med in medications:
            brand_name = med.get('brand_name', '')
            generic_name = med.get('generic_name', '')
            
            # Try matching both brand and generic names
            brand_matches = self.medicine_matcher.find_matches(brand_name) if brand_name else []
            generic_matches = self.medicine_matcher.find_matches(generic_name) if generic_name else []
            
            # Combine and deduplicate matches
            all_matches = brand_matches + generic_matches
            unique_matches = {m.name: m for m in all_matches}.values()
            
            results.append({
                "prescribed": {
                    "brand_name": brand_name,
                    "generic_name": generic_name,
                    "strength": med.get('strength', ''),
                    "dosage_form": med.get('dosage_form', ''),
                    "quantity": med.get('quantity', ''),
                    "instructions": med.get('instructions', '')
                },
                "matches": [
                    {
                        "name": match.name,
                        "similarity": match.similarity,
                        "phonetic_match": match.phonetic_match,
                        "details": match.original_data
                    }
                    for match in unique_matches
                ]
            })
        
        return {"results": results}

def main():
    # Initialize processor with medicine database
    processor = PrescriptionProcessor('A_Z_medicines_dataset_of_India.csv')
    
    # Process prescription image
    image_path = r'C:\\Users\\ihamz\\curestock\\paracetamol.jpg'
    
    # Extract text from prescription
    prescription_text = processor.extract_text_from_image(image_path)
    
    # Match medicines and get results
    results = processor.match_medicines_from_prescription(prescription_text)
    
    # Print results
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()