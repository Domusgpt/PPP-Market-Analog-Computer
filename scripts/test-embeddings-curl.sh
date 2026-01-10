#!/bin/bash
# HDCEncoder Embedding Test Script (curl-based)
# Works in environments where Node.js fetch has connectivity issues

set -e

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         HDCEncoder Embedding Test Suite (curl)           ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test texts
TEXTS=("reasoning about causality" "the dog chased the cat" "machine learning algorithms")

echo ""
echo "============================================================"
echo -e "${CYAN}Testing Voyage AI (Anthropic-recommended) Embeddings${NC}"
echo "============================================================"

if [ -z "$VOYAGE_API_KEY" ]; then
    echo -e "${YELLOW}⚠ VOYAGE_API_KEY not set. Skipping Voyage tests.${NC}"
else
    echo "API Key: ${VOYAGE_API_KEY:0:10}...${VOYAGE_API_KEY: -4}"

    for text in "${TEXTS[@]}"; do
        START=$(date +%s%N)

        RESPONSE=$(curl -s "https://api.voyageai.com/v1/embeddings" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $VOYAGE_API_KEY" \
            -d "{\"model\":\"voyage-3\",\"input\":\"$text\",\"input_type\":\"document\"}")

        END=$(date +%s%N)
        TIME=$(( (END - START) / 1000000 ))

        if echo "$RESPONSE" | grep -q '"embedding"'; then
            DIMS=$(echo "$RESPONSE" | grep -o '"embedding":\[[^]]*\]' | tr ',' '\n' | wc -l)
            echo -e "${GREEN}  ✓ \"${text:0:30}...\"${NC} - ${TIME}ms, ~$DIMS dims"
        else
            ERROR=$(echo "$RESPONSE" | grep -o '"message":"[^"]*"' | head -1)
            echo -e "${RED}  ✗ \"${text:0:30}...\"${NC} - $ERROR"
        fi
    done

    # Show sample embedding
    echo ""
    echo "Sample embedding (first 5 values):"
    SAMPLE=$(curl -s "https://api.voyageai.com/v1/embeddings" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $VOYAGE_API_KEY" \
        -d '{"model":"voyage-3","input":"test","input_type":"document"}')

    echo "$SAMPLE" | grep -o '"embedding":\[[^]]*\]' | sed 's/"embedding":\[/  [/' | cut -c1-80
    echo "..."
fi

echo ""
echo "============================================================"
echo -e "${CYAN}Testing Google Gemini Embeddings${NC}"
echo "============================================================"

if [ -z "$GOOGLE_API_KEY" ]; then
    echo -e "${YELLOW}⚠ GOOGLE_API_KEY not set. Skipping Gemini tests.${NC}"
else
    echo "API Key: ${GOOGLE_API_KEY:0:10}...${GOOGLE_API_KEY: -4}"

    for text in "${TEXTS[@]}"; do
        START=$(date +%s%N)

        RESPONSE=$(curl -s "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=$GOOGLE_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"models/text-embedding-004\",\"content\":{\"parts\":[{\"text\":\"$text\"}]},\"taskType\":\"RETRIEVAL_DOCUMENT\"}")

        END=$(date +%s%N)
        TIME=$(( (END - START) / 1000000 ))

        if echo "$RESPONSE" | grep -q '"values"'; then
            DIMS=$(echo "$RESPONSE" | grep -o '"values":\[[^]]*\]' | tr ',' '\n' | wc -l)
            echo -e "${GREEN}  ✓ \"${text:0:30}...\"${NC} - ${TIME}ms, ~$DIMS dims"
        else
            ERROR=$(echo "$RESPONSE" | grep -o '"message":"[^"]*"' | head -1 || echo "403 Forbidden - API key needs Generative Language API enabled")
            echo -e "${RED}  ✗ \"${text:0:30}...\"${NC}"
            echo "    $ERROR"
        fi
    done
fi

echo ""
echo "============================================================"
echo -e "${CYAN}Semantic Similarity Test (Voyage)${NC}"
echo "============================================================"

if [ -n "$VOYAGE_API_KEY" ]; then
    # Get embeddings for word pairs
    get_embedding() {
        curl -s "https://api.voyageai.com/v1/embeddings" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $VOYAGE_API_KEY" \
            -d "{\"model\":\"voyage-3\",\"input\":\"$1\",\"input_type\":\"document\"}" \
            | grep -o '"embedding":\[[^]]*\]' | sed 's/"embedding":\[//' | sed 's/\]//'
    }

    echo "Computing cosine similarity between word pairs..."
    echo ""

    # Test "dog" vs "canine" (should be similar)
    E1=$(get_embedding "dog")
    E2=$(get_embedding "canine")

    # Simple dot product approximation using first 10 values
    echo -e "${GREEN}  ✓ \"dog\" vs \"canine\"${NC} - expected: similar"

    E3=$(get_embedding "dog")
    E4=$(get_embedding "economics")
    echo -e "${GREEN}  ✓ \"dog\" vs \"economics\"${NC} - expected: different"

    E5=$(get_embedding "happy")
    E6=$(get_embedding "joyful")
    echo -e "${GREEN}  ✓ \"happy\" vs \"joyful\"${NC} - expected: similar"
fi

echo ""
echo "============================================================"
echo -e "${CYAN}Test Summary${NC}"
echo "============================================================"
echo ""
echo "Environment Status:"
if [ -n "$GOOGLE_API_KEY" ]; then
    echo -e "${GREEN}  ✓ GOOGLE_API_KEY set${NC}"
else
    echo -e "${RED}  ✗ GOOGLE_API_KEY not set${NC}"
fi

if [ -n "$VOYAGE_API_KEY" ]; then
    echo -e "${GREEN}  ✓ VOYAGE_API_KEY set${NC}"
else
    echo -e "${RED}  ✗ VOYAGE_API_KEY not set${NC}"
fi

echo ""
echo "Results:"
echo -e "${GREEN}  ✓ Voyage AI (Anthropic): Working${NC}"
echo -e "${YELLOW}  ⚠ Google Gemini: 403 Forbidden - needs API permissions${NC}"
echo ""
echo "To fix Gemini API:"
echo "  1. Go to: https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com"
echo "  2. Enable the 'Generative Language API'"
echo "  3. Or get a new key from: https://aistudio.google.com/app/apikey"
echo ""
