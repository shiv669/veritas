import json
import re

def process_text_file(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into paragraphs (more aggressive splitting)
    paragraphs = re.split(r'\n\s*\n|\n\d+\.\s+', content)
    
    # Clean and filter paragraphs
    cleaned_paragraphs = []
    for para in paragraphs:
        # Remove extra whitespace and newlines
        para = re.sub(r'\s+', ' ', para.strip())
        # Skip empty paragraphs or very short ones
        if len(para) > 100:  # Increased minimum length threshold
            cleaned_paragraphs.append(para)
    
    print(f"Found {len(cleaned_paragraphs)} paragraphs")
    
    # Create a list of documents
    documents = []
    for i, para in enumerate(cleaned_paragraphs):
        doc = {
            "id": f"doc_{i}",
            "content": para,
            "metadata": {
                "source": "1.json",
                "paragraph_index": i,
                "length": len(para)
            }
        }
        documents.append(doc)
    
    # Write to output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"documents": documents}, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(documents)} documents")
    print(f"Total characters: {sum(len(doc['content']) for doc in documents)}")

if __name__ == "__main__":
    process_text_file("data/1.json", "data/processed_1.json") 