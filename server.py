from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
import uuid
import uvicorn
import logging
from pathlib import Path
from typing import Optional
import asyncio
from contextlib import asynccontextmanager

# Import your pipeline
from survey_ai import SurveyAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting AI-Powered Data Analysis Server")
    yield
    logger.info("Shutting down server")

app = FastAPI(
    title="AI-Powered Data Analysis API",
    description="Upload survey data and get AI-powered analysis with downloadable reports",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Utility functions
def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not supported. Please upload CSV or Excel files."
        )

def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load DataFrame from file with error handling"""
    try:
        file_ext = Path(filepath).suffix.lower()
        if file_ext == '.csv':
            df = pd.read_csv(filepath, on_bad_lines="skip", encoding='utf-8')
        else:  # Excel files
            df = pd.read_excel(filepath)
        
        if df.empty:
            raise ValueError("The uploaded file is empty")
        
        logger.info(f"Successfully loaded DataFrame with shape: {df.shape}")
        return df
        
    except UnicodeDecodeError:
        # Try different encoding for CSV
        try:
            df = pd.read_csv(filepath, on_bad_lines="skip", encoding='latin-1')
            logger.info(f"Loaded CSV with latin-1 encoding, shape: {df.shape}")
            return df
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read file with any encoding: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return filepath"""
    # Generate unique filename
    unique_id = str(uuid.uuid4())[:8]
    file_ext = Path(file.filename).suffix.lower()
    safe_filename = f"{unique_id}_{file.filename.replace(' ', '_')}"
    filepath = os.path.join(UPLOAD_DIR, safe_filename)
    
    try:
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        with open(filepath, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved: {filepath} ({len(content)} bytes)")
        return filepath
        
    except Exception as e:
        # Clean up if save failed
        if os.path.exists(filepath):
            os.remove(filepath)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

def generate_output_paths(prefix: str) -> dict:
    """Generate output file paths"""
    unique_id = str(uuid.uuid4())[:8]
    return {
        'csv': os.path.join(OUTPUT_DIR, f"{prefix}_{unique_id}_cleaned.csv"),
        'html': os.path.join(OUTPUT_DIR, f"{prefix}_{unique_id}_report.html"),
        'pdf': os.path.join(OUTPUT_DIR, f"{prefix}_{unique_id}_report.pdf"),
        'csv_url': f"/download/{prefix}_cleaned_{unique_id}",
        'html_url': f"/download/{prefix}_html_{unique_id}",
        'pdf_url': f"/download/{prefix}_pdf_{unique_id}"
    }

def cleanup_old_files():
    """Clean up old uploaded and output files (called periodically)"""
    try:
        import time
        current_time = time.time()
        max_age = 3600  # 1 hour
        
        for directory in [UPLOAD_DIR, OUTPUT_DIR]:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    if current_time - os.path.getmtime(filepath) > max_age:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old file: {filepath}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Routes
@app.get("/")
async def root():
    """Redirect to main application"""
    return RedirectResponse(url="/static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Data Analysis API"}

@app.get("/analyze")
async def analyze_info():
    """Info about analyze endpoint"""
    return {
        "message": "This endpoint only supports POST with a file upload.",
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
        "usage": "Use the web interface at / to upload your dataset."
    }

@app.post("/analyze")
async def analyze_ai_mode(file: UploadFile = File(...)):
    """AI-powered automatic analysis"""
    logger.info(f"Starting AI analysis for file: {file.filename}")
    
    # Validate file
    validate_file(file)
    
    # Save file
    filepath = await save_uploaded_file(file)
    
    try:
        # Load data
        df = load_dataframe(filepath)
        logger.info(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        # Generate output paths
        output_paths = generate_output_paths('ai')
        
        # Run AI pipeline
        logger.info("Starting AI pipeline")
        ai = (SurveyAI(df)
              .impute_missing()
              .handle_outliers()
              .enforce_rules()
              .apply_weights()
              .stats_blocks()
              .visuals())
        
        # Save outputs
        ai.save_clean(output_paths['csv'])
        ai.save_html(output_paths['html'], title="AI-Powered Survey Analysis Report")
        ai.try_pdf(output_paths['html'], output_paths['pdf'])
        
        logger.info("AI analysis completed successfully")
        
        # Store file mapping for downloads
        app.state.file_mapping = getattr(app.state, 'file_mapping', {})
        base_id = output_paths['csv_url'].split('_')[-1]
        app.state.file_mapping[f"ai_cleaned_{base_id}"] = output_paths['csv']
        app.state.file_mapping[f"ai_html_{base_id}"] = output_paths['html']
        app.state.file_mapping[f"ai_pdf_{base_id}"] = output_paths['pdf']
        
        return {
            "status": "success",
            "message": "AI analysis completed successfully",
            "csv_url": output_paths['csv_url'],
            "html_url": output_paths['html_url'],
            "pdf_url": output_paths['pdf_url'],
            "rows_processed": len(df),
            "columns_processed": len(df.columns)
        }
        
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.post("/manual")
async def manual_analyze(
    file: UploadFile = File(...),
    cleaning: str = Form(...),
    analysis: str = Form(...)
):
    """Manual analysis with user-selected options"""
    logger.info(f"Starting manual analysis for file: {file.filename}")
    logger.info(f"Cleaning: {cleaning}, Analysis: {analysis}")
    
    # Validate inputs
    validate_file(file)
    
    valid_cleaning = {"missing", "outliers", "dropna"}
    valid_analysis = {"descriptives", "correlation", "topcats", "hist_box", "scatter", "full_report"}
    
    if cleaning not in valid_cleaning:
        raise HTTPException(status_code=400, detail=f"Invalid cleaning option: {cleaning}")
    if analysis not in valid_analysis:
        raise HTTPException(status_code=400, detail=f"Invalid analysis option: {analysis}")
    
    # Save file
    filepath = await save_uploaded_file(file)
    
    try:
        # Load data
        df = load_dataframe(filepath)
        logger.info(f"Loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
        
        # Generate output paths
        output_paths = generate_output_paths('manual')
        
        # Initialize AI pipeline
        ai = SurveyAI(df)
        
        # Apply selected cleaning strategy
        if cleaning == "missing":
            ai.impute_missing()
        elif cleaning == "outliers":
            ai.handle_outliers()
        elif cleaning == "dropna":
            df = df.dropna().reset_index(drop=True)
            ai.df = df
        
        # Apply selected analysis
        if analysis == "descriptives":
            ai.stats_blocks()
        elif analysis == "correlation":
            ai.stats_blocks().visuals(max_num=2)
        elif analysis == "topcats":
            ai.stats_blocks()
        elif analysis == "hist_box":
            ai.stats_blocks().visuals(max_num=3)
        elif analysis == "scatter":
            ai.stats_blocks().visuals(max_num=3)
        elif analysis == "full_report":
            ai.stats_blocks().visuals()
        
        # Save outputs
        ai.save_clean(output_paths['csv'])
        ai.save_html(output_paths['html'], title="Manual Analysis Report")
        ai.try_pdf(output_paths['html'], output_paths['pdf'])
        
        logger.info("Manual analysis completed successfully")
        
        # Store file mapping for downloads
        app.state.file_mapping = getattr(app.state, 'file_mapping', {})
        base_id = output_paths['csv_url'].split('_')[-1]
        app.state.file_mapping[f"manual_cleaned_{base_id}"] = output_paths['csv']
        app.state.file_mapping[f"manual_html_{base_id}"] = output_paths['html']
        app.state.file_mapping[f"manual_pdf_{base_id}"] = output_paths['pdf']
        
        return {
            "status": "success",
            "message": f"Manual analysis completed with {cleaning} cleaning and {analysis} analysis",
            "csv_url": output_paths['csv_url'],
            "html_url": output_paths['html_url'],
            "pdf_url": output_paths['pdf_url'],
            "rows_processed": len(ai.df),
            "columns_processed": len(ai.df.columns),
            "cleaning_applied": cleaning,
            "analysis_applied": analysis
        }
        
    except Exception as e:
        logger.error(f"Error in manual analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.get("/download/{file_key}")
async def download_file(file_key: str):
    """Download generated files"""
    file_mapping = getattr(app.state, 'file_mapping', {})
    
    if file_key not in file_mapping:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    filepath = file_mapping[file_key]
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File no longer available")
    
    # Determine filename and media type
    filename = os.path.basename(filepath)
    
    if filepath.endswith('.csv'):
        media_type = 'text/csv'
    elif filepath.endswith('.html'):
        media_type = 'text/html'
    elif filepath.endswith('.pdf'):
        media_type = 'application/pdf'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type=media_type,
        headers={"Cache-Control": "no-cache"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# Periodic cleanup task
@app.on_event("startup")
async def startup_event():
    """Setup periodic tasks"""
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # Run every hour
            cleanup_old_files()
    
    asyncio.create_task(periodic_cleanup())

if __name__ == "__main__":
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )