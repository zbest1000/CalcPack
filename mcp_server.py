#!/usr/bin/env python3
"""
MCP Server for PaddleOCR Integration
Implements Model Context Protocol server for OCR functionality
"""

import asyncio
import json
import base64
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import tempfile
import os

# MCP server imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        Resource,
        Prompt,
        GetPromptResult,
        CallToolResult,
        ListToolsResult,
        GetResourceResult,
        ListResourcesResult,
    )
except ImportError as e:
    logging.error(f"MCP server imports failed: {e}")
    logging.info("Please install MCP: pip install mcp")

# Local imports
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaddleOCRMCPServer:
    """MCP Server for PaddleOCR integration"""
    
    def __init__(self, paddleocr_service_url: str = "http://localhost:8000"):
        """Initialize MCP server"""
        self.service_url = paddleocr_service_url
        self.server = Server("paddleocr-mcp")
        self.session = requests.Session()
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available OCR tools"""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="ocr_text_detection_recognition",
                        description="Perform OCR (text detection and recognition) on an image",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data or image URL"
                                },
                                "lang": {
                                    "type": "string",
                                    "description": "Language code (default: ch for Chinese)",
                                    "default": "ch"
                                },
                                "use_doc_orientation_classify": {
                                    "type": "boolean",
                                    "description": "Use document orientation classification",
                                    "default": False
                                },
                                "use_doc_unwarping": {
                                    "type": "boolean",
                                    "description": "Use document unwarping",
                                    "default": False
                                },
                                "use_textline_orientation": {
                                    "type": "boolean",
                                    "description": "Use text line orientation classification",
                                    "default": False
                                }
                            },
                            "required": ["image_data"]
                        }
                    ),
                    Tool(
                        name="structure_analysis",
                        description="Perform document structure analysis (PP-StructureV3)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data or image URL"
                                },
                                "output_format": {
                                    "type": "string",
                                    "description": "Output format: json or markdown",
                                    "enum": ["json", "markdown"],
                                    "default": "json"
                                },
                                "use_seal_recognition": {
                                    "type": "boolean",
                                    "description": "Use seal recognition",
                                    "default": True
                                },
                                "use_table_recognition": {
                                    "type": "boolean",
                                    "description": "Use table recognition",
                                    "default": True
                                },
                                "use_formula_recognition": {
                                    "type": "boolean",
                                    "description": "Use formula recognition",
                                    "default": True
                                }
                            },
                            "required": ["image_data"]
                        }
                    ),
                    Tool(
                        name="text_detection_only",
                        description="Perform text detection only (without recognition)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data or image URL"
                                }
                            },
                            "required": ["image_data"]
                        }
                    ),
                    Tool(
                        name="text_recognition_only",
                        description="Perform text recognition only (without detection)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data or image URL"
                                },
                                "lang": {
                                    "type": "string",
                                    "description": "Language code (default: ch for Chinese)",
                                    "default": "ch"
                                }
                            },
                            "required": ["image_data"]
                        }
                    ),
                    Tool(
                        name="chat_ocr",
                        description="Intelligent document understanding with key information extraction",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data or image URL"
                                },
                                "key_list": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of keys to extract from the document"
                                },
                                "api_key": {
                                    "type": "string",
                                    "description": "API key for the chat model (required)"
                                }
                            },
                            "required": ["image_data", "key_list", "api_key"]
                        }
                    )
                ]
            )
            
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls"""
            try:
                if name == "ocr_text_detection_recognition":
                    return await self._handle_ocr_tool(arguments)
                elif name == "structure_analysis":
                    return await self._handle_structure_tool(arguments)
                elif name == "text_detection_only":
                    return await self._handle_detection_tool(arguments)
                elif name == "text_recognition_only":
                    return await self._handle_recognition_tool(arguments)
                elif name == "chat_ocr":
                    return await self._handle_chat_ocr_tool(arguments)
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Unknown tool: {name}")]
                    )
            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Tool execution failed: {str(e)}")]
                )
                
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List available resources"""
            return ListResourcesResult(
                resources=[
                    Resource(
                        uri="paddleocr://models/list",
                        name="Available PaddleOCR Models",
                        description="List of available PaddleOCR models and configurations",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="paddleocr://docs/api",
                        name="PaddleOCR API Documentation",
                        description="Complete API documentation for PaddleOCR integration",
                        mimeType="text/markdown"
                    )
                ]
            )
            
        @self.server.get_resource()
        async def get_resource(uri: str) -> GetResourceResult:
            """Get resource content"""
            if uri == "paddleocr://models/list":
                models_info = {
                    "ocr_models": {
                        "PP-OCRv5": {
                            "description": "Latest high-accuracy text recognition model",
                            "languages": ["ch", "en", "fr", "de", "ko", "ja"],
                            "features": ["text_detection", "text_recognition", "handwriting"]
                        },
                        "PP-OCRv4": {
                            "description": "Previous generation OCR model",
                            "languages": ["ch", "en"],
                            "features": ["text_detection", "text_recognition"]
                        }
                    },
                    "structure_models": {
                        "PP-StructureV3": {
                            "description": "Document structure analysis model",
                            "features": ["layout_detection", "table_recognition", "formula_recognition", "seal_recognition"]
                        }
                    },
                    "chat_models": {
                        "PP-ChatOCRv4": {
                            "description": "Intelligent document understanding model",
                            "features": ["key_information_extraction", "document_qa"]
                        }
                    }
                }
                
                return GetResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=json.dumps(models_info, indent=2, ensure_ascii=False)
                        )
                    ]
                )
                
            elif uri == "paddleocr://docs/api":
                api_docs = """# PaddleOCR MCP Server API Documentation

## Available Tools

### 1. ocr_text_detection_recognition
Performs complete OCR processing including text detection and recognition.

**Parameters:**
- `image_data` (required): Base64 encoded image data or image URL
- `lang` (optional): Language code, default "ch"
- `use_doc_orientation_classify` (optional): Enable document orientation classification
- `use_doc_unwarping` (optional): Enable document unwarping
- `use_textline_orientation` (optional): Enable text line orientation classification

### 2. structure_analysis
Performs document structure analysis using PP-StructureV3.

**Parameters:**
- `image_data` (required): Base64 encoded image data or image URL
- `output_format` (optional): "json" or "markdown", default "json"
- `use_seal_recognition` (optional): Enable seal recognition
- `use_table_recognition` (optional): Enable table recognition
- `use_formula_recognition` (optional): Enable formula recognition

### 3. text_detection_only
Performs text detection without recognition.

### 4. text_recognition_only
Performs text recognition without detection.

### 5. chat_ocr
Intelligent document understanding with key information extraction.

**Parameters:**
- `image_data` (required): Base64 encoded image data or image URL
- `key_list` (required): List of keys to extract
- `api_key` (required): API key for the chat model

## Usage Examples

```python
# OCR processing
result = await call_tool("ocr_text_detection_recognition", {
    "image_data": "base64_encoded_image",
    "lang": "ch"
})

# Structure analysis
result = await call_tool("structure_analysis", {
    "image_data": "base64_encoded_image",
    "output_format": "json"
})
```
"""
                
                return GetResourceResult(
                    contents=[TextContent(type="text", text=api_docs)]
                )
                
            else:
                raise ValueError(f"Unknown resource: {uri}")

    async def _prepare_image_data(self, image_input: str) -> str:
        """Prepare image data for processing"""
        # Check if it's a URL
        if image_input.startswith(('http://', 'https://')):
            try:
                response = self.session.get(image_input, timeout=30)
                response.raise_for_status()
                image_data = base64.b64encode(response.content).decode('utf-8')
                return image_data
            except Exception as e:
                raise ValueError(f"Failed to fetch image from URL: {e}")
        
        # Check if it's already base64 (remove data URL prefix if present)
        if image_input.startswith('data:image'):
            return image_input.split(',')[1]
        
        # Assume it's already base64 encoded
        return image_input

    async def _handle_ocr_tool(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle OCR tool call"""
        try:
            image_data = await self._prepare_image_data(arguments["image_data"])
            
            payload = {
                "image_data": image_data,
                "lang": arguments.get("lang", "ch"),
                "use_doc_orientation_classify": arguments.get("use_doc_orientation_classify", False),
                "use_doc_unwarping": arguments.get("use_doc_unwarping", False),
                "use_textline_orientation": arguments.get("use_textline_orientation", False)
            }
            
            response = self.session.post(
                f"{self.service_url}/ocr",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Format the result for display
            if result.get("status") == "success":
                texts = []
                for page_result in result.get("result", []):
                    if isinstance(page_result, dict) and "texts" in page_result:
                        for text_item in page_result["texts"]:
                            texts.append(f"Text: {text_item['text']} (Confidence: {text_item['confidence']:.3f})")
                
                formatted_result = f"OCR Results:\n" + "\n".join(texts)
                
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=formatted_result),
                        TextContent(type="text", text=f"\nRaw Result:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
                    ]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"OCR failed: {result}")]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"OCR processing failed: {str(e)}")]
            )

    async def _handle_structure_tool(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle structure analysis tool call"""
        try:
            image_data = await self._prepare_image_data(arguments["image_data"])
            
            payload = {
                "image_data": image_data,
                "output_format": arguments.get("output_format", "json"),
                "use_seal_recognition": arguments.get("use_seal_recognition", True),
                "use_table_recognition": arguments.get("use_table_recognition", True),
                "use_formula_recognition": arguments.get("use_formula_recognition", True)
            }
            
            response = self.session.post(
                f"{self.service_url}/structure",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", 
                            text=f"Structure Analysis Results:\n{json.dumps(result['result'], ensure_ascii=False, indent=2)}"
                        )
                    ]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Structure analysis failed: {result}")]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Structure analysis failed: {str(e)}")]
            )

    async def _handle_detection_tool(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle text detection tool call"""
        try:
            image_data = await self._prepare_image_data(arguments["image_data"])
            
            payload = {"image_data": image_data}
            
            response = self.session.post(
                f"{self.service_url}/text_detection",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                detection_results = result.get("result", [])
                summary = f"Found {len(detection_results)} text regions"
                
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=f"Text Detection Results:\n{summary}"),
                        TextContent(type="text", text=f"\nDetailed Results:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
                    ]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Text detection failed: {result}")]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Text detection failed: {str(e)}")]
            )

    async def _handle_recognition_tool(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle text recognition tool call"""
        try:
            image_data = await self._prepare_image_data(arguments["image_data"])
            
            payload = {
                "image_data": image_data,
                "lang": arguments.get("lang", "ch")
            }
            
            response = self.session.post(
                f"{self.service_url}/text_recognition",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text", 
                            text=f"Text Recognition Results:\n{json.dumps(result, ensure_ascii=False, indent=2)}"
                        )
                    ]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Text recognition failed: {result}")]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Text recognition failed: {str(e)}")]
            )

    async def _handle_chat_ocr_tool(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle chat OCR tool call"""
        try:
            image_data = await self._prepare_image_data(arguments["image_data"])
            
            payload = {
                "image_data": image_data,
                "key_list": arguments["key_list"],
                "api_key": arguments["api_key"]
            }
            
            response = self.session.post(
                f"{self.service_url}/chat_ocr",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                extracted_info = result.get("result", {})
                formatted_result = "Key Information Extraction Results:\n"
                
                for key in arguments["key_list"]:
                    value = extracted_info.get(key, "Not found")
                    formatted_result += f"- {key}: {value}\n"
                
                return CallToolResult(
                    content=[
                        TextContent(type="text", text=formatted_result),
                        TextContent(type="text", text=f"\nRaw Result:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
                    ]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Chat OCR failed: {result}")]
                )
                
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Chat OCR failed: {str(e)}")]
            )

    async def run(self):
        """Run the MCP server"""
        logger.info("Starting PaddleOCR MCP Server...")
        
        # Check if the PaddleOCR service is available
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"PaddleOCR service is available at {self.service_url}")
            else:
                logger.warning(f"PaddleOCR service returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to PaddleOCR service: {e}")
            logger.info("Make sure the PaddleOCR service is running")
        
        async with stdio_server() as streams:
            await self.server.run(
                streams[0], 
                streams[1], 
                self.server.create_initialization_options()
            )

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PaddleOCR MCP Server")
    parser.add_argument(
        "--service-url", 
        default="http://localhost:8000",
        help="URL of the PaddleOCR service"
    )
    
    args = parser.parse_args()
    
    server = PaddleOCRMCPServer(args.service_url)
    asyncio.run(server.run())

if __name__ == "__main__":
    main()