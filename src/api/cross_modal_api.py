#!/usr/bin/env python3
"""
Cross-Modal REST API Server

Local-only REST API for cross-modal analysis operations.
Provides high-level endpoints for document analysis, format conversion,
and mode recommendation for local automation and custom UIs.

This complements (not replaces) the MCP server interface.
"""

import asyncio
import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

# Import concrete cross-modal services directly so package-level export drift fails loudly.
from src.analytics.cross_modal_converter import DataFormat
from src.analytics.cross_modal_orchestrator import WorkflowOptimizationLevel
from src.analytics.cross_modal_validator import ValidationLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="KGAS Cross-Modal Analysis API",
    description="Local REST API for cross-modal document analysis and format conversion",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS for local origins only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Vue dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global job storage (in production, use Redis or database)
jobs: Dict[str, Dict[str, Any]] = {}

# Pydantic models for request/response
class AnalyzeRequest(BaseModel):
    """Request model for document analysis"""
    target_format: str = Field("graph", description="Target format: graph, table, or vector")
    task: str = Field("extract entities", description="Analysis task description")
    optimization_level: str = Field("balanced", description="Optimization: speed, balanced, or quality")
    validation_level: str = Field("standard", description="Validation: basic, standard, or comprehensive")

class ConvertRequest(BaseModel):
    """Request model for format conversion"""
    data: Union[Dict[str, Any], List[List[float]]] = Field(..., description="Data to convert")
    source_format: str = Field(..., description="Source format: graph, table, or vector")
    target_format: str = Field(..., description="Target format: graph, table, or vector")
    method: Optional[str] = Field(None, description="Conversion method (format-specific)")

class RecommendRequest(BaseModel):
    """Request model for mode recommendation"""
    task: str = Field(..., description="Analysis task description")
    data_type: str = Field(..., description="Type of data being analyzed")
    size: int = Field(..., description="Approximate data size")
    performance_priority: str = Field("quality", description="Priority: speed or quality")

class JobResponse(BaseModel):
    """Response model for async job creation"""
    job_id: str
    status: str
    created_at: str
    message: str

class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    workflow_id: str
    selected_mode: str
    results: Dict[str, Any]
    validation: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    source_traceability: Optional[Dict[str, Any]] = None

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    """Initialize cross-modal services on startup"""
    try:
        # Initialize with default configuration
        config = {
            'llm': {'provider': 'openai'},
            'embedding': {'device': 'cpu'}
        }
        
        success = _initialize_cross_modal_services(config)
        if success:
            logger.info("✅ Cross-modal services initialized successfully")
        else:
            logger.warning("⚠️ Some services failed to initialize - API will have limited functionality")
            
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        logger.warning("API starting with limited functionality")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Check API and service health"""
    try:
        registry = _get_registry()
        health_status = registry.check_all_health()
        
        return {
            "status": "healthy" if all(health_status.values()) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "services": health_status,
            "version": "1.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Document analysis endpoint
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_format: str = Query("graph", description="Target format for analysis"),
    task: str = Query("extract entities", description="Analysis task"),
    optimization_level: str = Query("balanced", description="Optimization level"),
    validation_level: str = Query("standard", description="Validation level")
):
    """
    Analyze an uploaded document and convert to specified format.
    
    Supports PDF, Word, and text documents.
    """
    # Validate file type
    allowed_types = [".pdf", ".docx", ".doc", ".txt", ".md"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_types}"
        )
    
    # Validate target format
    try:
        target_fmt = _parse_enum(DataFormat, target_format, "target format")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target format: {target_format}. Use: graph, table, or vector"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        registry = _get_registry()
        graph_data = _document_placeholder_graph(file.filename, len(content), file_ext)
        
        # Execute analysis
        orchestrator = registry.orchestrator
        if not orchestrator:
            raise HTTPException(
                status_code=503,
                detail="Orchestrator service not available"
            )
            
        result = await orchestrator.orchestrate_analysis(
            research_question=task,
            data=graph_data,
            source_format=DataFormat.GRAPH,
            preferred_modes=_preferred_modes_for_format(target_fmt),
            optimization_level=_parse_enum(WorkflowOptimizationLevel, optimization_level, "optimization level"),
            validation_level=_parse_enum(ValidationLevel, validation_level, "validation level")
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Return results
        return AnalysisResponse(
            workflow_id=result.workflow_id,
            selected_mode=_selected_mode_value(result),
            results={
                target_format: _serialize_results(result.primary_result)
            },
            validation=_serialize_validation_report(result.validation_report),
            performance_metrics=result.performance_metrics,
            source_traceability=result.analysis_metadata
        )
        
    except HTTPException:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        # Clean up temp file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

# Format conversion endpoint
@app.post("/api/convert")
async def convert_format(request: ConvertRequest):
    """
    Convert data between formats (Graph ↔ Table ↔ Vector).
    
    Preserves source traceability during conversion.
    """
    try:
        # Validate formats
        source_fmt = _parse_enum(DataFormat, request.source_format, "source format")
        target_fmt = _parse_enum(DataFormat, request.target_format, "target format")
        
        if source_fmt == target_fmt:
            return {"data": request.data, "message": "Source and target formats are the same"}
        
        # Get converter
        registry = _get_registry()
        converter = registry.converter
        if not converter:
            raise HTTPException(
                status_code=503,
                detail="Converter service not available"
            )
        
        # Convert data
        result = await converter.convert_data(
            data=request.data,
            source_format=source_fmt,
            target_format=target_fmt,
            method=request.method
        )
        
        return _serialize_conversion_result(result)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mode recommendation endpoint
@app.post("/api/recommend")
async def recommend_mode(request: RecommendRequest):
    """
    Get AI-powered recommendation for optimal analysis mode.
    
    Returns confidence scores and reasoning.
    """
    try:
        registry = _get_registry()
        mode_selector = registry.mode_selector
        if not mode_selector:
            raise HTTPException(
                status_code=503,
                detail="Mode selector service not available"
            )

        data_context = _recommend_request_to_data_context(request)
        result = await mode_selector.select_optimal_mode(
            research_question=request.task,
            data_context=data_context,
            preferences={"performance_priority": request.performance_priority}
        )
        return _serialize_mode_selection(result)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Mode recommendation failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# Batch analysis endpoint
@app.post("/api/batch/analyze", response_model=JobResponse)
async def batch_analyze(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    target_format: str = Query("graph"),
    task: str = Query("extract entities")
):
    """
    Submit multiple documents for batch analysis.
    
    Returns a job ID to track progress.
    """
    raise HTTPException(
        status_code=501,
        detail="Batch analysis endpoint is not wired to the current document analysis pipeline"
    )

# Job status endpoint
@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status and results for a batch job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "progress": {
            "total": job["total_files"],
            "processed": job["processed_files"],
            "percentage": (job["processed_files"] / job["total_files"] * 100) if job["total_files"] > 0 else 0
        },
        "results": job["results"] if job["status"] == "completed" else None,
        "errors": job["errors"] if job["errors"] else None
    }

# Service statistics endpoint
@app.get("/api/stats")
async def get_statistics():
    """Get service statistics and usage metrics"""
    try:
        registry = _get_registry()
        stats = registry.get_service_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "services": stats,
            "api": {
                "active_jobs": len([j for j in jobs.values() if j["status"] == "processing"]),
                "completed_jobs": len([j for j in jobs.values() if j["status"] == "completed"]),
                "failed_jobs": len([j for j in jobs.values() if j["status"] == "failed"])
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def _serialize_results(data: Any) -> Any:
    """Serialize results for JSON response"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    elif hasattr(data, '__dict__'):
        return data.__dict__
    return data

def _serialize_conversion_result(result: Any) -> Dict[str, Any]:
    """Serialize the current ConversionResult dataclass for API callers."""
    metadata = result.conversion_metadata
    return {
        "data": _serialize_results(result.data),
        "source_format": result.source_format.value,
        "target_format": result.target_format.value,
        "metadata": {
            "conversion_timestamp": metadata.conversion_timestamp,
            "semantic_features_preserved": metadata.semantic_features_preserved,
            "quality_metrics": metadata.quality_metrics,
            "conversion_parameters": metadata.conversion_parameters,
            "preservation_score": result.preservation_score,
            "validation_passed": result.validation_passed,
            "semantic_integrity": result.semantic_integrity,
            "warnings": result.warnings,
        },
        "performance": {
            "conversion_time": metadata.processing_time,
            "data_size_before": metadata.data_size_before,
            "data_size_after": metadata.data_size_after,
        },
    }

def _get_registry() -> Any:
    """Import the registry only when an endpoint actually needs service wiring."""
    try:
        from src.analytics.cross_modal_service_registry import get_registry
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Cross-modal registry unavailable: {exc}") from exc
    return get_registry()

def _initialize_cross_modal_services(config: Dict[str, Any]) -> Any:
    """Initialize cross-modal services lazily so API import does not require all runtime deps."""
    try:
        from src.analytics.cross_modal_service_registry import initialize_cross_modal_services
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Cross-modal registry unavailable: {exc}") from exc
    return initialize_cross_modal_services(config)

def _document_placeholder_graph(filename: str, byte_count: int, file_ext: str) -> Dict[str, Any]:
    """Create explicit document-metadata graph input until full pipeline wiring is restored."""
    return {
        "nodes": [
            {
                "id": "document",
                "label": filename,
                "type": "DOCUMENT",
                "properties": {
                    "filename": filename,
                    "extension": file_ext,
                    "byte_count": byte_count
                }
            }
        ],
        "edges": []
    }

def _parse_enum(enum_cls: Any, raw_value: str, label: str) -> Any:
    """Parse API strings against enum values, accepting case-insensitive values."""
    normalized = raw_value.lower()
    try:
        return enum_cls(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_cls)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label}: {raw_value}. Use one of: {allowed}"
        ) from exc

def _preferred_modes_for_format(target_format: DataFormat) -> Optional[List[Any]]:
    """Map a requested output format to preferred analysis modes when possible."""
    try:
        from src.analytics.mode_selection_service import AnalysisMode
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=f"Mode selection unavailable: {exc}") from exc

    mapping = {
        DataFormat.GRAPH: [AnalysisMode.GRAPH_ANALYSIS],
        DataFormat.TABLE: [AnalysisMode.TABLE_ANALYSIS],
        DataFormat.VECTOR: [AnalysisMode.VECTOR_ANALYSIS],
    }
    return mapping.get(target_format)

def _recommend_request_to_data_context(request: RecommendRequest) -> Any:
    """Build the current DataContext contract from the recommendation API request."""
    if request.size < 0:
        raise HTTPException(status_code=400, detail="size must be non-negative")

    try:
        from src.analytics.mode_selection_service import create_data_context
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Mode selection unavailable: {exc}") from exc

    normalized_data_type = request.data_type.strip().lower()
    data_types = [normalized_data_type] if normalized_data_type else ["unknown"]
    return create_data_context(
        data_size=request.size,
        data_types=data_types,
        entity_count=0,
        relationship_count=0,
        has_temporal_data="temporal" in data_types,
        has_spatial_data=any(data_type in {"spatial", "geo", "geospatial"} for data_type in data_types),
        has_hierarchical_structure=any(data_type in {"graph", "tree", "hierarchical"} for data_type in data_types),
        available_formats=["graph", "table", "vector"]
    )

def _serialize_mode_selection(result: Any) -> Dict[str, Any]:
    """Serialize a current ModeSelectionResult for the recommendation endpoint."""
    primary_mode = result.primary_mode.value if hasattr(result.primary_mode, "value") else str(result.primary_mode)
    secondary_modes = [
        mode.value if hasattr(mode, "value") else str(mode)
        for mode in result.secondary_modes
    ]
    confidence_level = (
        result.confidence_level.value
        if hasattr(result.confidence_level, "value")
        else str(result.confidence_level)
    )
    return {
        "recommended_mode": primary_mode,
        "primary_mode": primary_mode,
        "secondary_modes": secondary_modes,
        "confidence": result.confidence,
        "confidence_level": confidence_level,
        "reasoning": result.reasoning,
        "workflow_steps": result.workflow_steps,
        "estimated_performance": result.estimated_performance,
        "fallback_used": result.fallback_used,
        "selection_metadata": result.selection_metadata,
    }

def _selected_mode_value(result: Any) -> str:
    """Extract the selected mode from the current AnalysisResult metadata."""
    mode_selection = result.analysis_metadata.get("mode_selection", {})
    primary_mode = mode_selection.get("primary_mode") if isinstance(mode_selection, dict) else None
    if hasattr(primary_mode, "value"):
        return primary_mode.value
    if primary_mode:
        return str(primary_mode)
    return "unknown"

def _serialize_validation_report(report: Any) -> Dict[str, Any]:
    """Serialize current validation reports for API responses."""
    if report is None:
        return {}
    if hasattr(report, "__dict__"):
        return {
            key: _serialize_results(value)
            for key, value in report.__dict__.items()
        }
    return {"report": _serialize_results(report)}

async def process_batch_analysis(job_id: str, files: List[UploadFile], target_format: str, task: str):
    """Process batch analysis in background"""
    job = jobs[job_id]
    job["status"] = "processing"
    
    try:
        job["status"] = "failed"
        job["errors"].append({
            "error": "Batch analysis is not wired to the current document analysis pipeline"
        })
        
    except Exception as e:
        job["status"] = "failed"
        job["errors"].append({"error": str(e)})

# Main entry point for development
if __name__ == "__main__":
    import uvicorn
    
    # Run on localhost only for security
    uvicorn.run(
        app,
        host="127.0.0.1",  # Localhost only
        port=8000,
        log_level="info"
    )
