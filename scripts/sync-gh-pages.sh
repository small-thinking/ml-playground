#!/bin/bash

# Check if there are changes in the visualizations folder
if ! git diff --quiet HEAD~1 HEAD -- visualizations/; then
    echo "Changes detected in visualizations folder. Proceeding with sync..."
else
    echo "No changes detected in visualizations folder. Skipping sync."
    exit 0
fi

# Sync gh-pages branch with main branch
echo "Syncing gh-pages branch with main..."

# Create or update gh-pages branch
git checkout -B gh-pages
git reset --hard main

# Push to gh-pages branch
git push origin gh-pages --force

# Return to main branch
git checkout main

echo "gh-pages branch synced successfully!"
echo "Your site will be available at: https://username.github.io/ml-playground/" 
