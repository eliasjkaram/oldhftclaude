# Manual Upload Instructions for Google Drive Files

## 🎯 Quick Upload Process

Since I cannot directly access Google Drive links, here are the best methods to upload your files to MinIO:

### Method 1: Automated Script (Recommended)

I've created a Python script that will handle the entire process:

```bash
# Run the automated uploader
python google_drive_to_minio_uploader.py
```

This script will:
- ✅ Download all 33 files from Google Drive
- ✅ Upload them to your MinIO bucket
- ✅ Provide detailed progress reports
- ✅ Handle errors and retries

### Method 2: Manual Download + Upload

If the script doesn't work, here's the manual process:

#### Step 1: Download from Google Drive
For each link, convert the URL format:
- **From**: `https://drive.google.com/file/d/FILE_ID/view?usp=drive_link`
- **To**: `https://drive.google.com/uc?export=download&id=FILE_ID`

#### Step 2: Upload to MinIO
Use the MinIO client or Python script:

```python
from minio import Minio

client = Minio(
    'uschristmas.us',
    access_key='AKSTOCKDB2024',
    secret_key='StockDB-Secret-Access-Key-2024-Secure!',
    secure=True
)

# Upload file
client.fput_object('stockdb', 'uploaded_data/filename.csv', 'local_file.csv')
```

### Method 3: Google Drive API (Advanced)

If you have Google Drive API access:

```python
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Use Google Drive API to download files
# Then upload to MinIO using the script above
```

## 🔧 Troubleshooting

### Common Issues:

1. **Large File Downloads**
   - Google Drive may show a warning for large files
   - The script handles this automatically
   - For manual downloads, click "Download anyway"

2. **Rate Limiting**
   - Google Drive may limit download speed
   - The script includes delays between downloads
   - Be patient with large files

3. **MinIO Upload Errors**
   - Check your credentials in `.env`
   - Verify network connectivity
   - Ensure bucket permissions

### File ID Extraction

Your Google Drive URLs contain these file IDs:
```
1k0kjdAVtJdpFnu7FvrFon_jxnDXMKlCT
13khDULoN42ejJ63_6J-2aAcVLaMyqWHr
1-jyN6kSUKpNcuzMel1XEVsbmUijluC-J
... (and 30 more)
```

## 📊 Expected Results

After successful upload, you should have:
- ✅ **33 new files** in MinIO bucket
- ✅ **Files organized** in `uploaded_data/` directory
- ✅ **Detailed logs** of the upload process
- ✅ **Size verification** for each file

## 🚀 Next Steps

Once uploaded, you can:

1. **Verify uploads**:
```python
from minio_options_data_manager import MinIOOptionsDataManager
manager = MinIOOptionsDataManager()
# Check for new files in uploaded_data/ directory
```

2. **Process the data**:
```python
from comprehensive_data_pipeline import ComprehensiveDataPipeline
pipeline = ComprehensiveDataPipeline()
# Process the newly uploaded files
```

3. **Integrate with trading system**:
```python
from final_production_system import ProductionSystemManager
system = ProductionSystemManager()
# Include new data in trading algorithms
```

## ⚡ Ready to Upload

Run this command to start the automated upload:

```bash
python google_drive_to_minio_uploader.py
```

The script will handle everything automatically and provide detailed progress reports!