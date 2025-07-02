#!/usr/bin/env python3
"""
PaddleOCR Integration Test Script
Demonstrates various ways to use the PaddleOCR service
"""

import asyncio
import base64
import json
import requests
import time
from pathlib import Path
from typing import Dict, Any

# Test image data (small test image encoded as base64)
TEST_IMAGE_B64 = """
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==
"""

class PaddleOCRTester:
    """Test class for PaddleOCR service"""
    
    def __init__(self, service_url: str = "http://localhost:8000"):
        """Initialize tester with service URL"""
        self.service_url = service_url
        self.session = requests.Session()
        
    def check_service_health(self) -> bool:
        """Check if the service is running"""
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            return False
    
    def test_basic_ocr(self) -> Dict[str, Any]:
        """Test basic OCR functionality"""
        print("🔍 Testing basic OCR...")
        
        payload = {
            "image_data": TEST_IMAGE_B64.strip(),
            "lang": "ch",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False
        }
        
        try:
            response = self.session.post(
                f"{self.service_url}/ocr",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ OCR test completed: {result.get('status')}")
            return result
            
        except Exception as e:
            print(f"❌ OCR test failed: {e}")
            return {"error": str(e)}
    
    def test_text_detection(self) -> Dict[str, Any]:
        """Test text detection only"""
        print("📍 Testing text detection...")
        
        payload = {"image_data": TEST_IMAGE_B64.strip()}
        
        try:
            response = self.session.post(
                f"{self.service_url}/text_detection",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Text detection test completed: {result.get('status')}")
            return result
            
        except Exception as e:
            print(f"❌ Text detection test failed: {e}")
            return {"error": str(e)}
    
    def test_text_recognition(self) -> Dict[str, Any]:
        """Test text recognition only"""
        print("🔤 Testing text recognition...")
        
        payload = {
            "image_data": TEST_IMAGE_B64.strip(),
            "lang": "ch"
        }
        
        try:
            response = self.session.post(
                f"{self.service_url}/text_recognition",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Text recognition test completed: {result.get('status')}")
            return result
            
        except Exception as e:
            print(f"❌ Text recognition test failed: {e}")
            return {"error": str(e)}
    
    def test_structure_analysis(self) -> Dict[str, Any]:
        """Test document structure analysis"""
        print("📄 Testing structure analysis...")
        
        payload = {
            "image_data": TEST_IMAGE_B64.strip(),
            "output_format": "json",
            "use_seal_recognition": True,
            "use_table_recognition": True,
            "use_formula_recognition": True
        }
        
        try:
            response = self.session.post(
                f"{self.service_url}/structure",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Structure analysis test completed: {result.get('status')}")
            return result
            
        except Exception as e:
            print(f"❌ Structure analysis test failed: {e}")
            return {"error": str(e)}
    
    def test_file_upload(self, image_path: str) -> Dict[str, Any]:
        """Test file upload OCR"""
        print(f"📁 Testing file upload OCR with {image_path}...")
        
        if not Path(image_path).exists():
            print(f"❌ File not found: {image_path}")
            return {"error": "File not found"}
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'use_doc_orientation_classify': False,
                    'use_doc_unwarping': False,
                    'use_textline_orientation': False,
                    'lang': 'ch'
                }
                
                response = self.session.post(
                    f"{self.service_url}/upload_ocr",
                    files=files,
                    data=data,
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                
                print(f"✅ File upload test completed: {result.get('status')}")
                return result
                
        except Exception as e:
            print(f"❌ File upload test failed: {e}")
            return {"error": str(e)}
    
    def benchmark_performance(self, iterations: int = 5) -> Dict[str, Any]:
        """Benchmark OCR performance"""
        print(f"⏱️ Running performance benchmark ({iterations} iterations)...")
        
        times = []
        successful_requests = 0
        
        payload = {
            "image_data": TEST_IMAGE_B64.strip(),
            "lang": "ch"
        }
        
        for i in range(iterations):
            try:
                start_time = time.time()
                
                response = self.session.post(
                    f"{self.service_url}/ocr",
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                end_time = time.time()
                request_time = end_time - start_time
                times.append(request_time)
                successful_requests += 1
                
                print(f"  Request {i+1}: {request_time:.3f}s")
                
            except Exception as e:
                print(f"  Request {i+1}: Failed - {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            benchmark_result = {
                "total_requests": iterations,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / iterations * 100,
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "requests_per_second": 1 / avg_time if avg_time > 0 else 0
            }
            
            print(f"✅ Benchmark completed:")
            print(f"  Success rate: {benchmark_result['success_rate']:.1f}%")
            print(f"  Average time: {benchmark_result['average_time']:.3f}s")
            print(f"  Requests/sec: {benchmark_result['requests_per_second']:.2f}")
            
            return benchmark_result
        else:
            return {"error": "No successful requests"}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all available tests"""
        print("🧪 Running comprehensive PaddleOCR tests...\n")
        
        results = {}
        
        # Check service health
        if not self.check_service_health():
            print("❌ Service is not running. Please start the PaddleOCR service first.")
            return {"error": "Service not available"}
        
        print("✅ Service is running\n")
        
        # Run tests
        results["basic_ocr"] = self.test_basic_ocr()
        results["text_detection"] = self.test_text_detection()
        results["text_recognition"] = self.test_text_recognition()
        results["structure_analysis"] = self.test_structure_analysis()
        results["performance"] = self.benchmark_performance()
        
        return results

def create_sample_image():
    """Create a sample image for testing"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Create a simple test image
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((10, 30), "Hello PaddleOCR!", fill='black', font=font)
        draw.text((10, 60), "测试中文识别", fill='black', font=font)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Convert to base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_b64
        
    except ImportError:
        print("PIL not available, using simple test image")
        return TEST_IMAGE_B64.strip()

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PaddleOCR Service Tester")
    parser.add_argument("--url", default="http://localhost:8000", help="Service URL")
    parser.add_argument("--test", choices=["all", "ocr", "detection", "recognition", "structure", "performance"], 
                       default="all", help="Test to run")
    parser.add_argument("--image", help="Image file to test with")
    parser.add_argument("--iterations", type=int, default=5, help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = PaddleOCRTester(args.url)
    
    print(f"🚀 PaddleOCR Service Tester")
    print(f"Service URL: {args.url}")
    print(f"Test mode: {args.test}\n")
    
    # Run tests based on selection
    if args.test == "all":
        results = tester.run_all_tests()
    elif args.test == "ocr":
        results = {"ocr": tester.test_basic_ocr()}
    elif args.test == "detection":
        results = {"detection": tester.test_text_detection()}
    elif args.test == "recognition":
        results = {"recognition": tester.test_text_recognition()}
    elif args.test == "structure":
        results = {"structure": tester.test_structure_analysis()}
    elif args.test == "performance":
        results = {"performance": tester.benchmark_performance(args.iterations)}
    
    # Test file upload if image provided
    if args.image:
        results["file_upload"] = tester.test_file_upload(args.image)
    
    # Save results
    output_file = f"test_results_{int(time.time())}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 Results saved to: {output_file}")
    
    # Print summary
    print("\n📋 Test Summary:")
    for test_name, result in results.items():
        if isinstance(result, dict) and "error" not in result:
            status = result.get("status", "unknown")
            print(f"  {test_name}: ✅ {status}")
        else:
            print(f"  {test_name}: ❌ failed")

if __name__ == "__main__":
    main()