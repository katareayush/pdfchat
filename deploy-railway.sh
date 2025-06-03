#!/bin/bash

echo "🚂 Deploying PDF RAG to Railway"
echo "================================"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Login to Railway (if not already logged in)
echo "🔐 Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    echo "Please login to Railway:"
    railway login
fi

# Create or link project
echo "🔗 Setting up Railway project..."
if [ ! -f ".railway/project.json" ]; then
    echo "Creating new Railway project..."
    railway init
else
    echo "✅ Railway project already linked"
fi

# Deploy
echo "🚀 Deploying to Railway..."
railway up

# Get the deployment URL
echo "⏳ Getting deployment URL..."
sleep 10
DEPLOYMENT_URL=$(railway status --json | grep -o '"url":"[^"]*' | cut -d'"' -f4)

if [ ! -z "$DEPLOYMENT_URL" ]; then
    echo "✅ Deployment successful!"
    echo "🌐 Your PDF RAG app is live at: $DEPLOYMENT_URL"
    echo "📚 API documentation: $DEPLOYMENT_URL/docs"
    echo "❤️ Health check: $DEPLOYMENT_URL/health"
else
    echo "⚠️  Deployment completed, but URL not found. Check Railway dashboard."
fi

echo ""
echo "🎯 Next steps:"
echo "1. Visit your Railway dashboard to monitor the deployment"
echo "2. Check logs with: railway logs"
echo "3. Set environment variables if needed: railway env set KEY=value"
echo "4. Test your API endpoints"