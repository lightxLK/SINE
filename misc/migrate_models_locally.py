import os
import shutil
import glob

# Paths
CACHE_ROOT = r"C:\Users\Lokesh Patra\.cache\huggingface\hub"
PROJECT_MODELS = r"c:\Users\Lokesh Patra\Documents\Projects\CapstoneProject26\models"

# Map our internal folder names to the HF Cache folder names
model_maps = {
    "grammar_specialist": "models--pszemraj--grammar-synthesis-base",
    "bart_specialist": "models--facebook--bart-large-cnn",
    "flan_t5_specialist": "models--google--flan-t5-base"
}

def migrate_local_cache():
    for local_name, cache_name in model_maps.items():
        cache_path = os.path.join(CACHE_ROOT, cache_name, "snapshots")
        
        # Find the latest hash folder in the snapshots
        if not os.path.exists(cache_path):
            print(f"[Skip] {cache_name} not found in cache.")
            continue
            
        snapshots = glob.glob(os.path.join(cache_path, "*"))
        if not snapshots:
            print(f"[Skip] No snapshots for {cache_name}.")
            continue
            
        latest_snapshot = max(snapshots, key=os.path.getmtime)
        target_dir = os.path.join(PROJECT_MODELS, local_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        print(f"[Migrating] {cache_name} -> {target_dir}...")
        
        # Copy everything from the snapshot to the project
        for item in os.listdir(latest_snapshot):
            s = os.path.join(latest_snapshot, item)
            d = os.path.join(target_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d): shutil.rmtree(d)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        
        print(f"[Done] {local_name} migrated locally.")

if __name__ == "__main__":
    migrate_local_cache()
