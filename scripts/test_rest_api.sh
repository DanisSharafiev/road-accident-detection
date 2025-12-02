#!/bin/bash

# Test script for REST API endpoints
# Make sure the server is running before executing this script

SERVER_URL="http://localhost:8000"

echo "Testing Road Accident Detection REST API"
echo "============================================="
echo ""

# Test 1: Health Check
echo "Testing Health Check..."
curl -s "${SERVER_URL}/health" | python3 -m json.tool
echo ""
echo ""

# Test 2: Get Stats
echo "Testing Stats Endpoint..."
curl -s "${SERVER_URL}/api/stats" | python3 -m json.tool
echo ""
echo ""

# Test 3: Create Alert
echo "Testing Create Alert..."
curl -s -X POST "${SERVER_URL}/api/alerts" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-12-02T12:00:00",
    "severity": "critical",
    "message": "Test accident alert",
    "confidence": 0.95,
    "metadata": {"test": true}
  }' | python3 -m json.tool
echo ""
echo ""

# Test 4: Get All Alerts
echo "Testing Get Alerts..."
curl -s "${SERVER_URL}/api/alerts?limit=10" | python3 -m json.tool
echo ""
echo ""

# Test 5: Get Critical Alerts Only
echo "Testing Get Critical Alerts..."
curl -s "${SERVER_URL}/api/alerts?severity=critical&limit=5" | python3 -m json.tool
echo ""
echo ""

# Test 6: Root Endpoint
echo "Testing Root Endpoint..."
curl -s "${SERVER_URL}/" | python3 -m json.tool
echo ""
echo ""

echo "All tests completed!"
echo ""
echo "To view interactive API docs, visit:"
echo "   ${SERVER_URL}/docs"

