# AI-Powered Data Analysis System  

An interactive web application for analyzing survey or tabular datasets using **AI automation** or **manual control**. Users can upload CSV/Excel files, clean data, generate descriptive statistics, visualize insights, and export results in multiple formats (CSV, HTML, PDF).  

## 🚀 Features  

- **Web UI** with drag-and-drop file upload (CSV/Excel, max 50MB).  
- **Two analysis modes**:  
  - 🤖 **AI-Powered** – automated cleaning, outlier handling, smart analysis, full report.  
  - ⚙️ **Manual** – user selects cleaning strategy & analysis type for fine-grained control.  
- **Data Cleaning**: missing value imputation, outlier handling, rule enforcement.  
- **Analysis**: descriptive statistics, correlations, top categories, histograms, scatterplots, etc.  
- **Visuals**: interactive charts (histograms, boxplots, correlation heatmaps, categorical bars).  
- **Reports**: download cleaned dataset (`.csv`), HTML report, and PDF report.  
- **Backend**: FastAPI server with endpoints for AI and manual analysis.  
- **Automated Cleanup**: old uploaded/generated files removed periodically.  

## 🛠 Tech Stack  

- **Frontend**: HTML, CSS, JavaScript (Material Icons + custom styles).  
- **Backend**: FastAPI + Uvicorn.  
- **Data Processing**: Pandas, NumPy, SciPy.  
- **Visualization**: Matplotlib, Seaborn.  
- **Reporting**: HTML, PDF (via pdfkit/WeasyPrint/FPDF fallback).  

## 📂 Project Structure  
├── index.html # Frontend web UI
├── script.js # Frontend logic (upload, steps, mode selection)
├── style.css # Styling
├── server.py # FastAPI backend server
├── survey_ai.py # Core AI pipeline for data cleaning, analysis & reporting
├── uploads/ # Temporary uploaded files
├── outputs/ # Generated cleaned data & reports
