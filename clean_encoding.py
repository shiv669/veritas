import os
import json
import shutil
from pathlib import Path

def clean_processed_files(processed_dir):
    """Clean up encoding issues and remove macOS metadata files."""
    # Create backup directory
    backup_dir = os.path.join(processed_dir, 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    
    # Track statistics
    stats = {
        'total_files': 0,
        'removed_metadata': 0,
        'fixed_encoding': 0,
        'errors': 0
    }
    
    # Process each file
    for filename in os.listdir(processed_dir):
        if filename == 'backup':
            continue
            
        file_path = os.path.join(processed_dir, filename)
        stats['total_files'] += 1
        
        # Skip if it's a macOS metadata file
        if filename.startswith('._'):
            try:
                # Move to backup instead of deleting
                backup_path = os.path.join(backup_dir, filename)
                shutil.move(file_path, backup_path)
                stats['removed_metadata'] += 1
                print(f"Moved metadata file to backup: {filename}")
            except Exception as e:
                print(f"Error handling metadata file {filename}: {e}")
                stats['errors'] += 1
            continue
        
        # Try to fix encoding issues in regular files
        try:
            # First try reading with utf-8
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse JSON to validate
            json.loads(content)
            
        except UnicodeDecodeError:
            try:
                # Try different encodings
                encodings = ['latin1', 'cp1252', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        # If we can read it, try to parse JSON
                        json.loads(content)
                        # If successful, write back with utf-8
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        stats['fixed_encoding'] += 1
                        print(f"Fixed encoding for {filename} using {encoding}")
                        break
                    except:
                        continue
            except Exception as e:
                print(f"Error fixing encoding for {filename}: {e}")
                stats['errors'] += 1
                
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filename}")
            stats['errors'] += 1
            
        except Exception as e:
            print(f"Unexpected error processing {filename}: {e}")
            stats['errors'] += 1
    
    return stats

if __name__ == "__main__":
    processed_dir = "data/processed"
    stats = clean_processed_files(processed_dir)
    
    print("\nCleaning Results")
    print("=" * 50)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Metadata files removed: {stats['removed_metadata']}")
    print(f"Files with encoding fixed: {stats['fixed_encoding']}")
    print(f"Errors encountered: {stats['errors']}") 