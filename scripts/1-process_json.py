import json
import os

def process_json(input_file, output_file):
    """
    Process a JSON file for RAG:
    - If the file contains a top-level 'documents' list, extract it.
    - Otherwise, treat the entire file as a list of documents.
    Saves the resulting list to output_file in JSON format.
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load input JSON, support standard JSON or JSON Lines (one JSON object per line)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        # Fallback to JSON Lines format
        documents = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    documents.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")
        data = documents

    # Extract documents list
    if isinstance(data, dict) and 'documents' in data and isinstance(data['documents'], list):
        documents = data['documents']
    elif isinstance(data, list):
        documents = data
    else:
        # Wrap single document as list
        documents = [data]

    # Save processed JSON list
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(documents)} documents to {output_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process JSON file for RAG.")
    parser.add_argument('--input-file', '-i', required=True, help='Input JSON file')
    parser.add_argument('--output-file', '-o', required=True, help='Output processed JSON file')
    args = parser.parse_args()
    process_json(args.input_file, args.output_file) 