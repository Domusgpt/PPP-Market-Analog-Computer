#!/bin/bash
# Check GitHub Actions CI status for the current branch
# Usage: ./scripts/check-ci-status.sh [--wait]

set -e

REPO="Domusgpt/ppp-info-site"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
SHA=$(git rev-parse HEAD)

echo "Checking CI status for:"
echo "  Branch: $BRANCH"
echo "  Commit: ${SHA:0:7}"
echo ""

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Note: GITHUB_TOKEN not set. Using unauthenticated API (rate limited)."
    AUTH_HEADER=""
else
    AUTH_HEADER="-H \"Authorization: token $GITHUB_TOKEN\""
fi

check_status() {
    local response=$(curl -s \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/${REPO}/actions/runs?branch=${BRANCH}&per_page=5")

    # Check if we got a valid response
    if echo "$response" | grep -q '"message"'; then
        echo "API Error: $(echo "$response" | grep -o '"message":"[^"]*"')"
        return 1
    fi

    # Parse the most recent run
    local run_id=$(echo "$response" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)
    local status=$(echo "$response" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
    local conclusion=$(echo "$response" | grep -o '"conclusion":"[^"]*"' | head -1 | cut -d'"' -f4)
    local name=$(echo "$response" | grep -o '"name":"[^"]*"' | head -1 | cut -d'"' -f4)

    if [ -z "$run_id" ]; then
        echo "No workflow runs found for this branch."
        return 0
    fi

    echo "Latest run: $name"
    echo "  Run ID: $run_id"
    echo "  Status: $status"

    if [ "$status" = "completed" ]; then
        echo "  Conclusion: $conclusion"

        if [ "$conclusion" = "failure" ]; then
            echo ""
            echo "BUILD FAILED! Fetching logs..."
            echo ""

            # Get failed jobs
            local jobs_response=$(curl -s \
                -H "Accept: application/vnd.github.v3+json" \
                "https://api.github.com/repos/${REPO}/actions/runs/${run_id}/jobs")

            echo "$jobs_response" | grep -E '"name"|"conclusion"|"status"' | head -20

            echo ""
            echo "View full logs: https://github.com/${REPO}/actions/runs/${run_id}"
            return 1
        elif [ "$conclusion" = "success" ]; then
            echo ""
            echo "BUILD PASSED!"
            return 0
        fi
    else
        echo ""
        echo "Build still running..."
        return 2
    fi
}

if [ "$1" = "--wait" ]; then
    echo "Waiting for CI to complete..."
    while true; do
        check_status
        result=$?
        if [ $result -eq 0 ] || [ $result -eq 1 ]; then
            exit $result
        fi
        echo "Waiting 30 seconds..."
        sleep 30
    done
else
    check_status
fi
