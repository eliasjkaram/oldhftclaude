# MinIO S3 Endpoint Access Issues and Workaround

## Date: June 29, 2025

### Problem Description

During the development and enhancement of the 2025 options price prediction algorithm, persistent technical challenges were encountered when attempting to directly access historical stock and options data stored in the MinIO S3 bucket (`stockdb`) via its S3 endpoint.

The initial goal was to stream data directly from MinIO into Python scripts (`preprocess.ipynb`, `run_preprocess.py`) to avoid local storage and simplify the data pipeline.

### Tools and Errors Encountered

Multiple attempts were made using standard Python libraries for S3 interaction:

1.  **`minio` library:**
    *   Initial attempts to list and download objects resulted in unexpected behavior, such as printing `README.md` content instead of performing file operations.
    *   Further debugging revealed issues with the `download_prefix` function not correctly identifying or processing objects, leading to empty local directories.

2.  **`boto3` library (AWS SDK for Python):**
    *   Refactoring `minio_utils.py` to use `boto3` was attempted as an alternative.
    *   However, this led to `RecursionError` and `SyntaxError` within underlying `cffi` and `cryptography` dependencies, particularly when running in the current Python 3.13 virtual environment. These errors suggested deep-seated compatibility or configuration issues at a lower level of the software stack, preventing stable S3 client operation.

### Diagnosis

The recurring errors indicate a fundamental incompatibility or misconfiguration between the Python S3 client libraries (`minio`, `boto3`) and the specific environment or MinIO setup. Despite the MinIO endpoint being publicly accessible and credentials being correct, direct programmatic interaction proved unreliable. The errors were not typical access or connection issues, but rather internal library failures during execution.

### Workaround Implemented

Given the critical need to proceed with feature engineering and model training, and the inability to resolve the direct S3 access issues within the current environment, a workaround has been implemented:

*   **Manual Local Data Download:** The strategy has reverted to requiring manual download of the necessary historical stock and options data from the MinIO S3 bucket to the local filesystem.
*   **Modified Processing Scripts:**
    *   `run_preprocess.py` has been updated to remove all direct MinIO interaction code. It now expects the relevant CSV files to be present in the local `dataset/options/` and `dataset/stocks/` directories.
    *   The script reads data directly from these local paths using `pandas.read_csv()`.

### Impact and Future Considerations

*   **Immediate Progress:** This workaround allows immediate progress on the 2025 algorithm's development, enabling feature vector enhancement and model training.
*   **Manual Step:** It introduces a manual data synchronization step, which is less ideal for automation and scalability.
*   **Future Analysis:** For future iterations or deployment in different environments, further investigation into the `minio` and `boto3` compatibility issues with the specific Python version and underlying system libraries would be beneficial. This might involve:
    *   Testing with different Python versions.
    *   Exploring alternative S3 client libraries or custom HTTP requests if necessary.
    *   Consulting `minio` and `boto3` documentation/community for known issues related to the encountered errors.

This document serves as a record of the challenges faced and the pragmatic solution adopted to ensure project continuity.
