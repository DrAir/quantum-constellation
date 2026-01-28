
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.workflow.multi_doc import MultiDocWorkflow
    print("Import successful.")
    
    # Check if methods exist and are coroutines where expected
    import inspect
    
    if not inspect.iscoroutinefunction(MultiDocWorkflow._process_batch):
        print("Error: _process_batch is not async")
        sys.exit(1)
        
    if not hasattr(MultiDocWorkflow, '_process_single_contract'):
        print("Error: _process_single_contract method missing")
        sys.exit(1)
        
    if not inspect.iscoroutinefunction(MultiDocWorkflow._process_single_contract):
        print("Error: _process_single_contract is not async")
        sys.exit(1)
        
    print("Structure verification passed.")
    
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
