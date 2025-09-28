#!/bin/bash

# CLPO Project Page Deployment Script
# This script helps you deploy the CLPO project page to GitHub Pages

set -e

echo "🚀 CLPO Project Page Deployment Script"
echo "======================================"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository. Please run this script from your CLPO repository root."
    exit 1
fi

# Check if required files exist
echo "📁 Checking required files..."

required_files=(
    "docs/index.html"
    "docs/_config.yml" 
    "CLPO-teaser.jpg"
    "CLPO-workflow.png"
    "README.md"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Error: Required file $file not found!"
        exit 1
    fi
    echo "✅ Found: $file"
done

# Get user input for customization
echo ""
echo "🔧 Configuration Setup"
echo "====================="

read -p "Enter your GitHub username: " github_username
read -p "Enter your repository name (default: CLPO): " repo_name
repo_name=${repo_name:-CLPO}

read -p "Enter your paper arXiv ID (if available, or press enter to skip): " arxiv_id
read -p "Enter your Twitter handle (optional): " twitter_handle

# Update _config.yml
echo ""
echo "📝 Updating configuration files..."

sed -i.bak "s/your-username/$github_username/g" docs/_config.yml
sed -i.bak "s/your-twitter/$twitter_handle/g" docs/_config.yml

# Update README.md
sed -i.bak "s/your-username/$github_username/g" README.md

# Update index.html with links if provided
if [[ -n "$arxiv_id" ]]; then
    sed -i.bak "s|href=\"#\" class=\"link-btn\">.*<i class=\"fas fa-file-alt\"></i> Paper|href=\"https://arxiv.org/abs/$arxiv_id\" class=\"link-btn\"><i class=\"fas fa-file-alt\"></i> Paper|" docs/index.html
fi

if [[ -n "$github_username" ]]; then
    sed -i.bak "s|href=\"#\" class=\"link-btn\">.*<i class=\"fab fa-github\"></i> Code|href=\"https://github.com/$github_username/$repo_name\" class=\"link-btn\"><i class=\"fab fa-github\"></i> Code|" docs/index.html
fi

if [[ -n "$twitter_handle" ]]; then
    sed -i.bak "s|href=\"#\" class=\"link-btn\">.*<i class=\"fab fa-twitter\"></i> Twitter|href=\"https://twitter.com/$twitter_handle\" class=\"link-btn\"><i class=\"fab fa-twitter\"></i> Twitter|" docs/index.html
fi

# Clean up backup files
rm -f docs/_config.yml.bak README.md.bak docs/index.html.bak

echo "✅ Configuration files updated!"

# Git operations
echo ""
echo "📤 Preparing for deployment..."

# Add all changes
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit."
else
    # Commit changes
    commit_message="Deploy CLPO project page with GitHub Pages setup"
    git commit -m "$commit_message"
    echo "✅ Changes committed: $commit_message"
fi

# Push to remote
echo ""
echo "🔄 Pushing to GitHub..."
git push origin main

echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo "Your CLPO project page will be available at:"
echo "🌐 https://$github_username.github.io/$repo_name/"
echo ""
echo "📋 Next Steps:"
echo "1. Go to your GitHub repository settings"
echo "2. Navigate to the 'Pages' section"
echo "3. Under 'Source', select 'Deploy from a branch'"
echo "4. Choose 'main' branch and '/docs' folder"
echo "5. Click 'Save'"
echo ""
echo "⏱️  GitHub Pages deployment may take a few minutes to complete."
echo "📖 For detailed instructions, see docs/DEPLOYMENT.md"
echo ""
echo "🚀 Happy sharing!"
