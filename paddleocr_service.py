#!/usr/bin/env python3
"""
PaddleOCR Integration Service
Comprehensive OCR solution using PaddleOCR v3.x with MCP server support
"""

import os
import json
import base64
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import asyncio
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from paddleocr import PaddleOCR, PPStructureV3, PPChatOCRv4Doc
    from paddleocr import TextDetection, TextRecognition
except ImportError as e:
    logging.error(f"PaddleOCR import failed: {e}")
    logging.info("Please install PaddleOCR: pip install paddleocr")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCRRequest(BaseModel):
    """OCR request model"""
    image_data: str = Field(..., description="Base64 encoded image data")
    use_doc_orientation_classify: bool = Field(default=False, description="Use document orientation classification")
    use_doc_unwarping: bool = Field(default=False, description="Use document unwarping")
    use_textline_orientation: bool = Field(default=False, description="Use text line orientation classification")
    lang: str = Field(default="ch", description="Language code")
    ocr_version: str = Field(default="PP-OCRv5", description="OCR version to use")

class StructureRequest(BaseModel):
    """Structure analysis request model"""
    image_data: str = Field(..., description="Base64 encoded image data")
    use_doc_orientation_classify: bool = Field(default=False, description="Use document orientation classification")
    use_doc_unwarping: bool = Field(default=False, description="Use document unwarping")
    use_seal_recognition: bool = Field(default=True, description="Use seal recognition")
    use_table_recognition: bool = Field(default=True, description="Use table recognition")
    use_formula_recognition: bool = Field(default=True, description="Use formula recognition")
    output_format: str = Field(default="json", description="Output format: json, markdown")

class ChatOCRRequest(BaseModel):
    """Chat OCR request model"""
    image_data: str = Field(..., description="Base64 encoded image data")
    key_list: List[str] = Field(..., description="List of keys to extract")
    api_key: Optional[str] = Field(default=None, description="API key for chat model")
    use_mllm: bool = Field(default=False, description="Use multimodal large language model")

class PaddleOCRService:
    """Comprehensive PaddleOCR service implementation"""
    
    def __init__(self):
        """Initialize PaddleOCR service"""
        self.app = FastAPI(
            title="PaddleOCR Integration Service",
            description="Comprehensive OCR solution using PaddleOCR v3.x",
            version="3.1.0"
        )
        self._setup_routes()
        self._ocr_instances = {}
        self._structure_instance = None
        self._chat_ocr_instance = None
        
        # Initialize default OCR instance
        self._init_default_ocr()
        
    def _init_default_ocr(self):
        """Initialize default OCR instance"""
        try:
            self._ocr_instances['default'] = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang='ch'
            )
            logger.info("Default PaddleOCR instance initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize default OCR: {e}")
            
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "service": "PaddleOCR Integration Service",
                "version": "3.1.0",
                "endpoints": {
                    "ocr": "/ocr - Text detection and recognition",
                    "structure": "/structure - Document structure analysis",
                    "chat_ocr": "/chat_ocr - Intelligent document understanding",
                    "text_detection": "/text_detection - Text detection only",
                    "text_recognition": "/text_recognition - Text recognition only",
                    "health": "/health - Service health check"
                }
            }
            
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "PaddleOCR"}
            
        @self.app.post("/ocr")
        async def ocr_endpoint(request: OCRRequest):
            return await self.process_ocr(request)
            
        @self.app.post("/structure")
        async def structure_endpoint(request: StructureRequest):
            return await self.process_structure(request)
            
        @self.app.post("/chat_ocr")
        async def chat_ocr_endpoint(request: ChatOCRRequest):
            return await self.process_chat_ocr(request)
            
        @self.app.post("/text_detection")
        async def text_detection_endpoint(request: OCRRequest):
            return await self.process_text_detection(request)
            
        @self.app.post("/text_recognition") 
        async def text_recognition_endpoint(request: OCRRequest):
            return await self.process_text_recognition(request)
            
        @self.app.post("/upload_ocr")
        async def upload_ocr_endpoint(
            file: UploadFile = File(...),
            use_doc_orientation_classify: bool = Form(False),
            use_doc_unwarping: bool = Form(False),
            use_textline_orientation: bool = Form(False),
            lang: str = Form("ch")
        ):
            return await self.process_upload_ocr(
                file, use_doc_orientation_classify, use_doc_unwarping, 
                use_textline_orientation, lang
            )

    def _decode_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image data to numpy array"""
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
                
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
                
            return image
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")

    def _get_ocr_instance(self, config: Dict[str, Any]) -> PaddleOCR:
        """Get or create OCR instance with specific configuration"""
        config_key = json.dumps(config, sort_keys=True)
        
        if config_key not in self._ocr_instances:
            try:
                self._ocr_instances[config_key] = PaddleOCR(**config)
                logger.info(f"Created new OCR instance with config: {config}")
            except Exception as e:
                logger.error(f"Failed to create OCR instance: {e}")
                return self._ocr_instances.get('default')
                
        return self._ocr_instances[config_key]

    async def process_ocr(self, request: OCRRequest) -> Dict[str, Any]:
        """Process OCR request"""
        try:
            # Decode image
            image = self._decode_image(request.image_data)
            
            # Get OCR instance
            ocr_config = {
                'use_doc_orientation_classify': request.use_doc_orientation_classify,
                'use_doc_unwarping': request.use_doc_unwarping,
                'use_textline_orientation': request.use_textline_orientation,
                'lang': request.lang
            }
            
            ocr = self._get_ocr_instance(ocr_config)
            
            # Process image
            result = ocr.predict(image)
            
            # Format result
            formatted_result = self._format_ocr_result(result)
            
            return {
                "status": "success",
                "message": "OCR processing completed",
                "result": formatted_result
            }
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

    async def process_structure(self, request: StructureRequest) -> Dict[str, Any]:
        """Process document structure analysis"""
        try:
            # Initialize structure instance if needed
            if self._structure_instance is None:
                self._structure_instance = PPStructureV3(
                    use_doc_orientation_classify=request.use_doc_orientation_classify,
                    use_doc_unwarping=request.use_doc_unwarping
                )
                
            # Decode image
            image = self._decode_image(request.image_data)
            
            # Process image
            result = self._structure_instance.predict(input=image)
            
            # Format result
            formatted_result = self._format_structure_result(result, request.output_format)
            
            return {
                "status": "success",
                "message": "Structure analysis completed",
                "result": formatted_result,
                "output_format": request.output_format
            }
            
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Structure analysis failed: {str(e)}")

    async def process_chat_ocr(self, request: ChatOCRRequest) -> Dict[str, Any]:
        """Process intelligent document understanding"""
        try:
            if not request.api_key:
                raise ValueError("API key is required for ChatOCR")
                
            # Initialize ChatOCR instance if needed
            if self._chat_ocr_instance is None:
                chat_bot_config = {
                    "module_name": "chat_bot",
                    "model_name": "ernie-3.5-8k",
                    "base_url": "https://qianfan.baidubce.com/v2",
                    "api_type": "openai",
                    "api_key": request.api_key,
                }
                
                retriever_config = {
                    "module_name": "retriever",
                    "model_name": "embedding-v1",
                    "base_url": "https://qianfan.baidubce.com/v2",
                    "api_type": "qianfan",
                    "api_key": request.api_key,
                }
                
                self._chat_ocr_instance = PPChatOCRv4Doc(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False
                )
                
            # Decode image
            image = self._decode_image(request.image_data)
            
            # Process visual prediction
            visual_predict_res = self._chat_ocr_instance.visual_predict(
                input=image,
                use_common_ocr=True,
                use_seal_recognition=True,
                use_table_recognition=True,
            )
            
            # Extract visual information
            visual_info_list = []
            for res in visual_predict_res:
                visual_info_list.append(res["visual_info"])
                
            # Build vector and chat
            vector_info = self._chat_ocr_instance.build_vector(
                visual_info_list, 
                flag_save_bytes_vector=True,
                retriever_config=retriever_config
            )
            
            chat_result = self._chat_ocr_instance.chat(
                key_list=request.key_list,
                visual_info=visual_info_list,
                vector_info=vector_info,
                mllm_predict_info=None,
                chat_bot_config=chat_bot_config,
                retriever_config=retriever_config,
            )
            
            return {
                "status": "success",
                "message": "ChatOCR processing completed",
                "result": chat_result,
                "extracted_keys": request.key_list
            }
            
        except Exception as e:
            logger.error(f"ChatOCR processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"ChatOCR processing failed: {str(e)}")

    async def process_text_detection(self, request: OCRRequest) -> Dict[str, Any]:
        """Process text detection only"""
        try:
            # Decode image
            image = self._decode_image(request.image_data)
            
            # Initialize text detection model
            text_detection = TextDetection()
            
            # Process image
            result = text_detection.predict(image)
            
            # Format result
            formatted_result = self._format_detection_result(result)
            
            return {
                "status": "success",
                "message": "Text detection completed",
                "result": formatted_result
            }
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Text detection failed: {str(e)}")

    async def process_text_recognition(self, request: OCRRequest) -> Dict[str, Any]:
        """Process text recognition only"""
        try:
            # Decode image
            image = self._decode_image(request.image_data)
            
            # Initialize text recognition model
            text_recognition = TextRecognition(lang=request.lang)
            
            # Process image
            result = text_recognition.predict(image)
            
            # Format result
            formatted_result = self._format_recognition_result(result)
            
            return {
                "status": "success",
                "message": "Text recognition completed", 
                "result": formatted_result
            }
            
        except Exception as e:
            logger.error(f"Text recognition failed: {e}")
            raise HTTPException(status_code=500, detail=f"Text recognition failed: {str(e)}")

    async def process_upload_ocr(
        self, 
        file: UploadFile, 
        use_doc_orientation_classify: bool,
        use_doc_unwarping: bool,
        use_textline_orientation: bool,
        lang: str
    ) -> Dict[str, Any]:
        """Process uploaded file OCR"""
        try:
            # Read file content
            content = await file.read()
            
            # Convert to base64
            image_data = base64.b64encode(content).decode('utf-8')
            
            # Create request
            request = OCRRequest(
                image_data=image_data,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                lang=lang
            )
            
            # Process OCR
            return await self.process_ocr(request)
            
        except Exception as e:
            logger.error(f"Upload OCR failed: {e}")
            raise HTTPException(status_code=500, detail=f"Upload OCR failed: {str(e)}")

    def _format_ocr_result(self, result: List[Dict]) -> List[Dict]:
        """Format OCR result for API response"""
        formatted_results = []
        
        for res in result:
            if hasattr(res, 'res') and 'rec_texts' in res.res:
                # Extract text and confidence scores
                texts = res.res['rec_texts']
                scores = res.res.get('rec_scores', [])
                boxes = res.res.get('rec_boxes', [])
                
                text_results = []
                for i, text in enumerate(texts):
                    text_result = {
                        'text': text,
                        'confidence': float(scores[i]) if i < len(scores) else 0.0
                    }
                    if i < len(boxes):
                        text_result['bbox'] = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else boxes[i]
                    text_results.append(text_result)
                
                formatted_results.append({
                    'texts': text_results,
                    'page_index': res.res.get('page_index'),
                    'input_path': res.res.get('input_path')
                })
            else:
                # Fallback formatting
                formatted_results.append(str(res))
                
        return formatted_results

    def _format_structure_result(self, result: List[Dict], output_format: str) -> Dict:
        """Format structure analysis result"""
        try:
            if not result:
                return {"error": "No structure analysis result"}
                
            res = result[0]
            if hasattr(res, 'res'):
                structure_data = res.res
                
                formatted_result = {
                    'layout_detection': structure_data.get('layout_det_res', {}),
                    'ocr_results': structure_data.get('overall_ocr_res', {}),
                    'page_index': structure_data.get('page_index'),
                    'input_path': structure_data.get('input_path'),
                    'model_settings': structure_data.get('model_settings', {})
                }
                
                if output_format == "markdown":
                    # Convert to markdown if supported
                    try:
                        if hasattr(res, 'save_to_markdown'):
                            # This would save to file, we'll return a placeholder
                            formatted_result['markdown_note'] = "Markdown conversion available via save_to_markdown method"
                    except:
                        pass
                        
                return formatted_result
            else:
                return {"raw_result": str(res)}
                
        except Exception as e:
            logger.error(f"Failed to format structure result: {e}")
            return {"error": f"Failed to format result: {str(e)}"}

    def _format_detection_result(self, result: List[Dict]) -> List[Dict]:
        """Format text detection result"""
        formatted_results = []
        
        for res in result:
            if hasattr(res, 'res'):
                detection_data = {
                    'boxes': res.res.get('dt_polys', []),
                    'scores': res.res.get('dt_scores', []),
                    'input_path': res.res.get('input_path'),
                    'page_index': res.res.get('page_index')
                }
                
                # Convert numpy arrays to lists for JSON serialization
                if 'boxes' in detection_data and hasattr(detection_data['boxes'], 'tolist'):
                    detection_data['boxes'] = detection_data['boxes'].tolist()
                    
                formatted_results.append(detection_data)
            else:
                formatted_results.append(str(res))
                
        return formatted_results

    def _format_recognition_result(self, result: List[Dict]) -> List[Dict]:
        """Format text recognition result"""
        formatted_results = []
        
        for res in result:
            if hasattr(res, 'res'):
                recognition_data = {
                    'text': res.res.get('rec_text', ''),
                    'confidence': res.res.get('rec_score', 0.0),
                    'input_path': res.res.get('input_path'),
                    'page_index': res.res.get('page_index')
                }
                formatted_results.append(recognition_data)
            else:
                formatted_results.append(str(res))
                
        return formatted_results

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the FastAPI service"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port, **kwargs)

# MCP Server Implementation
class MCPOCRServer:
    """MCP Server for PaddleOCR integration"""
    
    def __init__(self, paddleocr_service_url: str = "http://localhost:8000"):
        self.service_url = paddleocr_service_url
        self.session = requests.Session()
        
    async def handle_ocr_request(self, image_data: str, **kwargs) -> Dict[str, Any]:
        """Handle OCR request via MCP"""
        try:
            response = self.session.post(
                f"{self.service_url}/ocr",
                json={
                    "image_data": image_data,
                    **kwargs
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"MCP OCR request failed: {str(e)}"}
            
    async def handle_structure_request(self, image_data: str, **kwargs) -> Dict[str, Any]:
        """Handle structure analysis request via MCP"""
        try:
            response = self.session.post(
                f"{self.service_url}/structure",
                json={
                    "image_data": image_data,
                    **kwargs
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"MCP structure request failed: {str(e)}"}

# CLI Interface
def create_cli():
    """Create CLI interface for PaddleOCR service"""
    import click
    
    @click.group()
    def cli():
        """PaddleOCR Integration Service CLI"""
        pass
        
    @cli.command()
    @click.option('--host', default='0.0.0.0', help='Host to bind to')
    @click.option('--port', default=8000, help='Port to bind to')
    @click.option('--workers', default=1, help='Number of worker processes')
    def serve(host, port, workers):
        """Start the PaddleOCR service"""
        service = PaddleOCRService()
        service.run(host=host, port=port, workers=workers)
        
    @cli.command()
    @click.argument('image_path')
    @click.option('--lang', default='ch', help='Language code')
    @click.option('--output', default='result.json', help='Output file')
    def ocr(image_path, lang, output):
        """Process image with OCR"""
        try:
            # Load image and convert to base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                
            # Initialize service
            service = PaddleOCRService()
            
            # Create request
            request = OCRRequest(
                image_data=image_data,
                lang=lang
            )
            
            # Process
            import asyncio
            result = asyncio.run(service.process_ocr(request))
            
            # Save result
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            click.echo(f"OCR result saved to {output}")
            
        except Exception as e:
            click.echo(f"OCR processing failed: {e}", err=True)
            
    return cli

if __name__ == "__main__":
    # Run as CLI or service
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        service = PaddleOCRService()
        service.run()
    else:
        cli = create_cli()
        cli()