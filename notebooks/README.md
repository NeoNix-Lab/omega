# Notebooks Directory

This folder is dedicated to interactive exploratory research, visualizations, and hypothesis prototyping.

### Best Practices:
1. **Clear Outputs Before Committing**: To keep the git repository lightweight, clear notebook outputs before saving and committing (`Kernel -> Restart and Clear All Outputs`).
2. **Import from Core**: Use `from quant_platform.access import DataGateway` rather than hardcoding local parquet file paths.
3. **Graduate to Scripts**: Once an exploration yields promising results, translate the logic into a parameterized Python script inside `../studies/` for reproducible execution.
