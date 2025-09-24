import asyncio
import json
import sys
import os
import aiohttp
from typing import Dict, Any, List

# Configuration - point to your existing FastAPI server
FASTAPI_SERVER_URL = "http://localhost:8000"  # Adjust if needed

class SOPMCPServer:
    def __init__(self):
        self.tools = [
            {
                "name": "sop.search",
                "description": "Search SOP documents using hybrid BM25 and dense vector search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "sop.read",
                "description": "Read a specific SOP section by section_id",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "section_id": {
                            "type": "string",
                            "description": "Section ID to read"
                        }
                    },
                    "required": ["section_id"]
                }
            },
            {
                "name": "web.search",
                "description": "Search the web for additional information (requires SERPER_API_KEY)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "cite.validate",
                "description": "Validate citation coverage in text",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to validate citations for"
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of citations"
                        }
                    },
                    "required": ["text", "citations"]
                }
            }
        ]
    
    async def _call_fastapi_endpoint(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper method to call FastAPI endpoints"""
        url = f"{FASTAPI_SERVER_URL}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=30) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        return {"error": f"HTTP {response.status}: {error_text}"}
        except aiohttp.ClientError as e:
            return {"error": f"Connection error: {str(e)}"}
        except asyncio.TimeoutError:
            return {"error": "Request timeout"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    async def sop_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search SOP documents by calling FastAPI endpoint"""
        data = {"query": query, "k": k}
        result = await self._call_fastapi_endpoint("/tool/sop.search", data)
        
        if "error" in result:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error calling SOP search: {result['error']}"
                    }
                ]
            }
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }
            ]
        }
    
    async def sop_read(self, section_id: str) -> Dict[str, Any]:
        """Read a specific SOP section by calling FastAPI endpoint"""
        data = {"section_id": section_id}
        result = await self._call_fastapi_endpoint("/tool/sop.read", data)
        
        if "error" in result:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error calling SOP read: {result['error']}"
                    }
                ]
            }
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }
            ]
        }
    
    async def web_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search the web by calling FastAPI endpoint"""
        data = {"query": query, "k": k}
        result = await self._call_fastapi_endpoint("/tool/web.search", data)
        
        if "error" in result:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error calling web search: {result['error']}"
                    }
                ]
            }
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }
            ]
        }
    
    async def cite_validate(self, text: str, citations: List[str]) -> Dict[str, Any]:
        """Validate citation coverage by calling FastAPI endpoint"""
        data = {"text": text, "citations": citations}
        result = await self._call_fastapi_endpoint("/tool/cite.validate", data)
        
        if "error" in result:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error calling cite validate: {result['error']}"
                    }
                ]
            }
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }
            ]
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC request"""
        method = request.get('method')
        params = request.get('params', {})
        request_id = request.get('id')
        
        try:
            if method == 'tools/list':
                result = {"tools": self.tools}
                
            elif method == 'tools/call':
                tool_name = params.get('name')
                arguments = params.get('arguments', {})
                
                if tool_name == 'sop.search':
                    result = await self.sop_search(
                        arguments.get('query'),
                        arguments.get('k', 5)
                    )
                elif tool_name == 'sop.read':
                    result = await self.sop_read(arguments.get('section_id'))
                elif tool_name == 'web.search':
                    result = await self.web_search(
                        arguments.get('query'),
                        arguments.get('k', 5)
                    )
                elif tool_name == 'cite.validate':
                    result = await self.cite_validate(
                        arguments.get('text'),
                        arguments.get('citations', [])
                    )
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
                    
            elif method == 'initialize':
                result = {
                    "protocolVersion": "2025-03-26",
                    "serverInfo": {"name": "SOP MCP Proxy Server", "version": "1.0.0"},
                    "capabilities": {
                        "tools": {}
                    }
                }
                # Send the required notification after responding
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }), flush=True)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }
    
    async def run(self):
        """Run the MCP server"""
        print("SOP MCP Proxy Server running on stdio", file=sys.stderr)
        print(f"Proxying requests to: {FASTAPI_SERVER_URL}", file=sys.stderr)
        
        try:
            while True:
                # Read line from stdin with better error handling
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, sys.stdin.readline
                    )
                    
                    # Check for EOF or empty input
                    if not line or line.strip() == "":
                        break
                    
                    # Parse JSON-RPC request
                    request = json.loads(line.strip())
                    
                    # Handle request
                    response = await self.handle_request(request)
                    
                    # Send response
                    print(json.dumps(response), flush=True)
                    
                except json.JSONDecodeError as e:
                    # Only send error if we actually got input
                    if line and line.strip():
                        error_response = {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {
                                "code": -32700,
                                "message": f"Parse error: {str(e)}"
                            }
                        }
                        print(json.dumps(error_response), flush=True)
                    
                except EOFError:
                    break
                    
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)

async def main():
    server = SOPMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())
