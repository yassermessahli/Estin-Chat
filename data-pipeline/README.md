# Data Pipeline - Document Processing & Transformation

Complete data processing pipeline for extracting, cleaning, and chunking educational documents using AI models.

## Quick Start

> **Note**: The complete pipeline orchestrator (`orchestrate.py`) is currently not working properly. Please use the individual components separately as described below.

### 1. Data Organization & Preprocessing

```bash
# From repo root - organize files by format and rename with standardized convention
python -m data-pipeline.data-precleaning-organization.scripts.organize_by_format
python -m data-pipeline.data-precleaning-organization.scripts.rename_files
python -m data-pipeline.data-precleaning-organization.scripts.generate_index
```

### 2. PDF Content Extraction

```bash
# From repo root - extract content from PDF files
python -c "
from data_pipeline.utils.load.pdf_loader import PDFLoader
loader = PDFLoader('path/to/your/file.pdf')
pages_data = loader.extract_all_pages()
print(f'Extracted {len(pages_data)} pages')
"
```

### 3. Content Cleaning (AI-Powered)

```bash
# From repo root - clean text content
python -c "
from data_pipeline.utils.transform.text_cleanup import TextCleanup
from data_pipeline.utils.transform.model import Model, ModelParams
model = Model(ModelParams(model='qwen3:4b'))
cleaner = TextCleanup(text='your_text_here', model=model)
cleaned_text = cleaner.process()
"
```

### 4. Content Chunking

```bash
# From repo root - split content into chunks
python -c "
from data_pipeline.utils.split.hierarchical_splitter import HierarchicalSplitter
splitter = HierarchicalSplitter(chunk_size=300, chunk_overlap=60)
chunks = splitter.split_text('your_cleaned_text_here')
print(f'Created {len(chunks)} chunks')
"
```

### 5. Complete Pipeline Orchestrator (Currently Not Working)

```bash
# This is currently not functional - use individual steps above instead
# python -m data-pipeline.pipeline.orchestrate
```

## Folder Structure

### `pipeline/` - Main Orchestration

**Purpose**: Coordinate the complete data processing workflow from PDF extraction to final chunked output.

**Files**:

- `orchestrate.py` - Main pipeline orchestrator with AI model integration ⚠️ **Currently not working**
- `test_orchestrate.py` - Unit tests for pipeline functionality

**Usage**:

```bash
# Pipeline orchestrator is currently not functional
# Use individual components from utils/ folder instead

# Test pipeline components (when fixed)
python -m data-pipeline.pipeline.test_orchestrate
```

### `utils/` - Core Processing Components

**Purpose**: Modular utilities for document loading, content splitting, and AI-powered transformation.

#### `utils/load/` - Document Loading

**Files**:

- `pdf_loader.py` - PDF content extraction (text, tables, images)
- `tests/` - Unit tests for PDF loading functionality

#### `utils/split/` - Content Chunking

**Files**:

- `hierarchical_splitter.py` - Intelligent text chunking with overlap
- `tests/` - Chunking algorithm tests

#### `utils/transform/` - AI-Powered Content Cleaning

**Files**:

- `text_cleanup.py` - AI text cleaning and enhancement
- `table_cleanup.py` - Table data cleaning and structuring
- `image_cleanup.py` - Image description generation
- `model.py` - AI model interface and configuration
- `prompts.py` - Prompt templates for AI models
- `batch_cleanup/` - Batch processing utilities
- `tests/` - Transformation tests

**Usage**:

```bash
# Individual component usage
python -c "from data_pipeline.utils.load.pdf_loader import PDFLoader; loader = PDFLoader('file.pdf')"
python -c "from data_pipeline.utils.split.hierarchical_splitter import HierarchicalSplitter; splitter = HierarchicalSplitter()"
python -c "from data_pipeline.utils.transform.text_cleanup import TextCleanup; cleaner = TextCleanup(text, model)"
```

### `data-precleaning-organization/` - Data Preparation

**Purpose**: Organize, rename, and prepare raw document collections for processing.

**Files**:

- `scripts/organize_by_format.py` - Sort files by format (PDF, DOCX, images, etc.)
- `scripts/rename_files.py` - Standardize filenames with academic metadata
- `scripts/generate_index.py` - Create file inventory and metadata index
- `scripts/move_irrelevant_files.py` - Filter out non-academic content
- `scripts/restore_ignored_files.py` - Restore accidentally filtered files
- `scripts/scrapping_drive_links.py` - Extract documents from drive links
- `scripts/load.py` - Bulk file loading utilities

**Usage**:

```bash
# Organize data collection by file type
python -m data-pipeline.data-precleaning-organization.scripts.organize_by_format

# Apply standardized naming convention (LEVEL_SEMESTER_MODULE_TYPE_YEAR)
python -m data-pipeline.data-precleaning-organization.scripts.rename_files

# Generate comprehensive file index
python -m data-pipeline.data-precleaning-organization.scripts.generate_index

# Filter irrelevant files
python -m data-pipeline.data-precleaning-organization.scripts.move_irrelevant_files
```

## Configuration

The pipeline uses specialized AI models for different content types:

- **Text Model**: `qwen3:4b` (text cleaning and enhancement)
- **Table Model**: `qwen3:4b` (table structure and data cleaning)
- **Vision Model**: `granite3.2-vision:latest` (image description generation)

## Pipeline Workflow

1. **Data Organization**: Sort and rename files with academic metadata
2. **PDF Loading**: Extract text, tables, and images from documents
3. **AI Transformation**: Clean and enhance content using specialized models
4. **Hierarchical Splitting**: Create intelligent chunks with metadata
5. **Output Generation**: Save raw, cleaned, and chunked data

## Output Structure

The pipeline generates multiple output formats:

- `raw_extracted/` - Original PDF extraction data
- `cleaned/` - AI-enhanced and cleaned content
- `chunked/` - Final chunks with rich metadata
- `final/` - Complete processed documents with statistics

## Metadata Extraction

Automatic extraction from filename patterns like:
`1CP_S1_ELEC_COURS_2022_CHAPITRE1.pdf` → `{level: "1CP", semester: "S1", module: "ELEC", type: "COURS", year: 2022}`

## Sample Usage

```bash
# Step-by-step workflow (since orchestrator is not working)

# 1. Organize and prepare data
python -m data-pipeline.data-precleaning-organization.scripts.organize_by_format
python -m data-pipeline.data-precleaning-organization.scripts.rename_files

# 2. Extract content from PDFs (manual Python scripting required)
# 3. Clean content with AI models (manual Python scripting required)
# 4. Split into chunks (manual Python scripting required)

# Complete orchestrator (not working currently):
# python -m data-pipeline.pipeline.orchestrate
```

## Current Status

- ✅ **Data Organization Scripts**: Fully functional
- ✅ **Individual Components**: PDF loader, splitter, cleaners work independently
- ❌ **Pipeline Orchestrator**: Currently not working - use components separately
- ✅ **AI Model Integration**: Text, table, and image cleaning models functional when used individually
