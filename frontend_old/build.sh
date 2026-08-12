#!/bin/bash
# Vercel Build Script for Vanilla JS Frontend
# Injects the backend API URL dynamically during deployment

echo "Building frontend for Vercel deployment..."

if [ -n "$NEXT_PUBLIC_API_BASE" ]; then
    echo "Injecting NEXT_PUBLIC_API_BASE: $NEXT_PUBLIC_API_BASE"
    # Replace the default API_BASE initialization with the Vercel environment variable
    sed -i "s|let API_BASE = window.API_BASE || \"\";|let API_BASE = \"$NEXT_PUBLIC_API_BASE\";|g" app.js
else
    echo "Warning: NEXT_PUBLIC_API_BASE is not set. API calls will default to relative paths."
fi

echo "Build complete."
