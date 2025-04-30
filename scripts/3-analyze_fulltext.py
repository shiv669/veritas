import json
import os
from collections import defaultdict

def analyze_fulltext_field(processed_dir):
    """Analyze the fullText field across all processed documents."""
    stats = {
        'total_docs': 0,
        'has_fulltext': 0,
        'null_fulltext': 0,
        'length_distribution': defaultdict(int),
        'sample_docs': {
            'with_text': [],
            'without_text': []
        }
    }
    
    # Walk through all files in the processed directory
    for filename in os.listdir(processed_dir):
        # Skip hidden files (e.g., ._ files)
        if filename.startswith('.'):
            continue
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(processed_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # If file contains a list of docs, iterate through each; otherwise wrap single doc
            docs = data if isinstance(data, list) else [data]

            for idx, doc in enumerate(docs):
                stats['total_docs'] += 1
                fulltext = doc.get('fullText')
                if fulltext:
                    stats['has_fulltext'] += 1
                    text_length = len(fulltext)
                    
                    # Categorize length into buckets
                    if text_length < 1000:
                        bucket = '< 1K'
                    elif text_length < 5000:
                        bucket = '1K-5K'
                    elif text_length < 10000:
                        bucket = '5K-10K'
                    elif text_length < 50000:
                        bucket = '10K-50K'
                    else:
                        bucket = '> 50K'
                    
                    stats['length_distribution'][bucket] += 1
                    
                    # Record sample filename with index
                    if len(stats['sample_docs']['with_text']) < 5:
                        stats['sample_docs']['with_text'].append(f"{filename}[{idx}]")
                else:
                    stats['null_fulltext'] += 1
                    if len(stats['sample_docs']['without_text']) < 5:
                        stats['sample_docs']['without_text'].append(f"{filename}[{idx}]")
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return stats

if __name__ == "__main__":
    processed_dir = "data/processed"
    stats = analyze_fulltext_field(processed_dir)
    
    # Print results
    print("\nFullText Field Analysis")
    print("=" * 50)
    print(f"Total documents analyzed: {stats['total_docs']}")
    print(f"Documents with fullText: {stats['has_fulltext']} ({(stats['has_fulltext']/stats['total_docs']*100):.1f}%)")
    print(f"Documents without fullText: {stats['null_fulltext']} ({(stats['null_fulltext']/stats['total_docs']*100):.1f}%)")
    
    print("\nLength Distribution of fullText:")
    for bucket, count in sorted(stats['length_distribution'].items()):
        print(f"{bucket}: {count} documents ({(count/stats['has_fulltext']*100):.1f}%)")
    
    print("\nSample Documents with fullText:")
    for doc in stats['sample_docs']['with_text']:
        print(f"- {doc}")
        
    print("\nSample Documents without fullText:")
    for doc in stats['sample_docs']['without_text']:
        print(f"- {doc}") 