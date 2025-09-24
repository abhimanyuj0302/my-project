#!/usr/bin/env python3
import json
import subprocess
import sys

def test_mcp_server_simple():
    """Simple synchronous test"""

    # Use the same Python interpreter that's running this script
    python_exe = sys.executable

    # Test commands
    tests = [
        {
            "name": "List Tools",
            "request": {"jsonrpc":"2.0","id":1,"method":"tools/list"}
        },
        {
            "name": "SOP Search",
            "request": {"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"sop.search","arguments":{"query":"safety procedures","k":3}}}
        },
        {
            "name": "SOP Read", 
            "request": {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"sop.read","arguments":{"section_id":"SOP-001::PROCEDURE"}}}
        },
        {
            "name": "Cite Validate",
            "request": {"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"cite.validate","arguments":{"text":"This is a test sentence. Another sentence with citation SOP-001::PROCEDURE.","citations":["SOP-001::PROCEDURE"]}}}
        }
    ]
    
    for test in tests:
        print(f"\n🧪 Testing: {test['name']}")
        print("-" * 40)
        
        # Use sys.executable instead of 'python'
        process = subprocess.Popen(
            [python_exe, 'mcp_server.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Send request and close stdin
        request_json = json.dumps(test['request'])
        stdout, stderr = process.communicate(input=request_json + '\n')
        
        # Print server startup message
        if stderr:
            print(f"Server: {stderr.strip()}")
        
        # Parse and display response
        if stdout.strip():
            try:
                response = json.loads(stdout.strip())
                if 'result' in response:
                    print("✅ Success!")
                    if test['name'] == 'List Tools':
                        tools = response['result']['tools']
                        for tool in tools:
                            print(f"   - {tool['name']}: {tool['description']}")
                    else:
                        content = response['result']['content'][0]['text']
                        data = json.loads(content)
                        print(f"   Response: {json.dumps(data, indent=2)}")
                else:
                    print(f"❌ Error: {response.get('error', 'Unknown error')}")
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response: {stdout}")
        else:
            print("❌ No response received")

if __name__ == "__main__":
    test_mcp_server_simple()