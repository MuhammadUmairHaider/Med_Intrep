import inspect

import trl
from trl import SFTConfig, SFTTrainer

print(f"TRL version: {trl.__version__}")

print("\nSFTConfig signature:")
try:
    print(inspect.signature(SFTConfig))
except Exception as e:
    print(f"Could not get signature for SFTConfig: {e}")

print("\nSFTTrainer signature:")
try:
    print(inspect.signature(SFTTrainer))
except Exception as e:
    print(f"Could not get signature for SFTTrainer: {e}")

print("\nSFTConfig mro:")
print(SFTConfig.mro())
