import json
import os

def process_json_file(input_file, output_dir):
    """Process a file containing multiple JSON documents and split into individual files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the file line by line
    with open(input_file, 'r', encoding='utf-8') as f:
        line_num = 0
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            try:
                # Parse JSON document
                doc = json.loads(line)
                
                # Create a unique filename based on ID or title
                filename = f"doc_{line_num}.json"
                if doc.get('coreId'):
                    filename = f"doc_{doc['coreId']}.json"
                elif doc.get('title'):
                    # Create filename from title (first 50 chars, alphanumeric only)
                    safe_title = ''.join(c for c in doc['title'] if c.isalnum())[:50]
                    filename = f"doc_{safe_title}.json"
                
                # Write individual document to file
                output_file = os.path.join(output_dir, filename)
                with open(output_file, 'w', encoding='utf-8') as out:
                    json.dump(doc, out, indent=2, ensure_ascii=False)
                
                line_num += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue

if __name__ == "__main__":
    input_file = "data/1.json"
    output_dir = "data/processed"
    process_json_file(input_file, output_dir)
    print("Processing complete!") 