#!/bin/bash
# Quick test to check if Gemini API is accessible from current environment

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GOOGLE_API_KEY not set"
    exit 1
fi

echo "Testing Gemini 3 Pro API access..."
echo "Model: gemini-3-pro-preview"
echo ""

response=$(curl -s -w "\n%{http_code}" -X POST \
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-preview:generateContent?key=${GOOGLE_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{"contents":[{"parts":[{"text":"Respond with exactly: GEMINI_OK"}]}]}')

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

echo "HTTP Status: $http_code"
echo ""

if [ "$http_code" = "200" ]; then
    echo "SUCCESS: Gemini API is accessible!"
    echo ""
    echo "Response:"
    echo "$body" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('candidates',[{}])[0].get('content',{}).get('parts',[{}])[0].get('text','No text'))" 2>/dev/null || echo "$body" | head -c 300
elif [ "$http_code" = "403" ]; then
    echo "BLOCKED: Google is blocking this IP range (403 Forbidden)"
    echo ""
    echo "This typically happens on:"
    echo "  - Cloud VPS providers"
    echo "  - Some data center IPs"
    echo ""
    echo "Solutions:"
    echo "  1. Run locally on your machine"
    echo "  2. Use GitHub Actions (may work)"
    echo "  3. Use Vertex AI with service account"
elif [ "$http_code" = "400" ]; then
    echo "ERROR: Bad request (check API key format)"
    echo "$body" | head -c 300
elif [ "$http_code" = "401" ] || [ "$http_code" = "403" ]; then
    echo "ERROR: Authentication failed (invalid API key?)"
    echo "$body" | head -c 300
else
    echo "ERROR: Unexpected response"
    echo "$body" | head -c 500
fi
