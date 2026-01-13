#!/bin/bash
# Production start script for LangGraph on Render

echo "Starting LangGraph server on port $PORT"
exec langgraph dev --port $PORT --host 0.0.0.0 --no-browser
