import asyncio
import json
import sys
import os
import pickle
import numpy as np
import faiss
import networkx as nx
from networkx.readwrite import gpickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import re

INDEX_DIR = "indexes"
DOC_META_PATH = os.path.join(INDEX_DIR, "doc_meta.pkl")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")
KG_PATH = os.path.join(INDEX_DIR, "kg.gpickle")
MODEL_NAME = "all-MiniLM-L6-v2"

# Global variables for loaded resources
doc_meta = []
faiss_index = None
bm25 = None
kg = None
model = None

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
        
        # Load resources on initialization
        self._load_resources()
    
    def _load_resources(self):
        """Load prebuilt indexes and models"""
        global doc_meta, faiss_index, bm25, kg, model
        
        if not os.path.exists(FAISS_PATH):
            raise RuntimeError("Indices not built. Run indexer.py first.")

        try:
            with open(DOC_META_PATH, "rb") as f:
                doc_meta = pickle.load(f)
            faiss_index = faiss.read_index(FAISS_PATH)
            with open(BM25_PATH, "rb") as f:
                bm25 = pickle.load(f)
            kg = gpickle.read_gpickle(KG_PATH)
            model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to load resources: {str(e)}")
    
    def _get_section_text(self, section_id):
        """Helper: get section text by section_id"""
        for m in doc_meta:
            if m["section_id"] == section_id:
                return m, m.get("section_name"), m.get("title")
        return None, None, None
    
    def sop_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search SOP documents"""
        # BM25 candidates
        tokens = query.split()
        bm25_scores = bm25.get_scores(tokens)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:k]
        bm25_candidates = [(doc_meta[i]["section_id"], float(bm25_scores[i]))
                       for i in bm25_top_idx if bm25_scores[i] > 0]

        # FAISS dense retrieval
        emb = model.encode([query])[0].astype("float32")
        faiss.normalize_L2(emb.reshape(1, -1))
        D, I = faiss_index.search(np.array([emb]), k)
        dense_cands = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            sid = doc_meta[idx]["section_id"]
            dense_cands.append((sid, float(score)))

        # Merge (simple): prefer dense then bm25 if not dup
        merged = {}
        for sid, score in dense_cands:
            merged[sid] = {"sid": sid, "dense_score": score, "bm25_score": 0.0}
        for sid, score in bm25_candidates:
            if sid in merged:
                merged[sid]["bm25_score"] = score
            else:
                merged[sid] = {"sid": sid, "dense_score": 0.0, "bm25_score": score}

        # Rank by combined score
        results = sorted(merged.values(), key=lambda x: (
            x["dense_score"]*0.7 + x["bm25_score"]*0.3), reverse=True)
        
        # return excerpts (first 300 chars)
        out = []
        for r in results[:k]:
            meta = next((m for m in doc_meta if m["section_id"] == r["sid"]), None)
            excerpt = ""
            if meta:
                # fetch node text from KG if available
                node = kg.nodes.get(r["sid"], {})
                excerpt = node.get("text", "")[:300]
            out.append({"section_id": r["sid"], "excerpt": excerpt, "scores": {
                       "dense": r["dense_score"], "bm25": r["bm25_score"]}})
        
        # also add KG neighbors for context
        for item in out:
            neighbors = list(kg.successors(item["section_id"]))[:3]
            item["neighbors"] = neighbors
            
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"query": query, "results": out}, indent=2)
                }
            ]
        }
    
    def sop_read(self, section_id: str) -> Dict[str, Any]:
        """Read a specific SOP section"""
        node = kg.nodes.get(section_id)
        if not node:
            return {
                "content": [
                    {
                        "type": "text", 
                        "text": f"Error: section_id '{section_id}' not found"
                    }
                ]
            }
            
        result_data = {
            "section_id": section_id, 
            "text": node.get("text", ""), 
            "title": node.get("title"), 
            "sop": node.get("sop"), 
            "section": node.get("section")
        }
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result_data, indent=2)
                }
            ]
        }
    
    def web_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search the web"""
        SERPER_KEY = os.environ.get("SERPER_API_KEY")
        if SERPER_KEY:
            import requests
            headers = {"X-API-KEY": SERPER_KEY}
            payload = {"q": query, "num": k}
            try:
                resp = requests.post(
                    "https://google.serper.dev/search", json=payload, headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for r in data.get("organic", [])[:k]:
                        results.append({"title": r.get("title"), "url": r.get(
                            "link"), "snippet": r.get("snippet")})
                    result_data = {"query": query, "results": results}
                else:
                    result_data = {"query": query, "results": [], "warning": "Serper returned non-200"}
            except Exception as e:
                result_data = {"query": query, "results": [], "warning": f"Search error: {str(e)}"}
        else:
            result_data = {"query": query, "results": [], "warning": "No SERPER_API_KEY configured; web.search is disabled in this environment."}
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result_data, indent=2)
                }
            ]
        }
    
    def cite_validate(self, text: str, citations: List[str]) -> Dict[str, Any]:
        """Validate citation coverage"""
        # sentence splitting (naive)
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        anchors = set()
        for c in citations:
            anchors.add(c)
        
        # a sentence is considered covered if it contains a known anchor (SOP-xxx or url) or if any citation provided maps to the topical sop section
        coverage = []
        for s in sentences:
            found = False
            for a in anchors:
                if a.lower() in s.lower() or a.startswith("http") and a in s:
                    found = True
                    break
            coverage.append({"sentence": s, "covered": found})
        
        # compute coverage %
        cov_pct = sum(
            1 for c in coverage if c["covered"]) / max(1, len(coverage)) * 100
        
        result_data = {"coverage_pct": cov_pct, "details": coverage}
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result_data, indent=2)
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
                    result = self.sop_search(
                        arguments.get('query'),
                        arguments.get('k', 5)
                    )
                elif tool_name == 'sop.read':
                    result = self.sop_read(arguments.get('section_id'))
                elif tool_name == 'web.search':
                    result = self.web_search(
                        arguments.get('query'),
                        arguments.get('k', 5)
                    )
                elif tool_name == 'cite.validate':
                    result = self.cite_validate(
                        arguments.get('text'),
                        arguments.get('citations', [])
                    )
                else:
                    raise ValueError(f"Unknown tool: {tool_name}")
                    
            elif method == 'initialize':
                result = {
                    "protocolVersion": "2025-03-26",
                    "serverInfo": {"name": "SOP MCP Server", "version": "1.0.0"},
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
        print("SOP MCP Server running on stdio", file=sys.stderr)
        
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