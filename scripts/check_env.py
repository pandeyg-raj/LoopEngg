import importlib.util as u

MODS = [
    "datasets",
    "huggingface_hub",
    "tiktoken",
    "transformers",
    "pyarrow",
    "pandas",
    "numpy",
    "matplotlib",
    "requests",
]

for m in MODS:
    status = "OK" if u.find_spec(m) else "MISSING"
    print(f"{m:20} {status}")
