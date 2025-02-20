#!/bin/bash

# Check if a path argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-file-or-directory>"
  exit 1
fi

# Set variables
SOURCE_PATH="$1"

REPO_DIR="hockey-en-mallorca/src/weights/"  # Change this to your actual repo path
BASENAME=$(basename "$SOURCE_PATH")


cd ..

# Copy the file/directory to the repository
cp -r "$SOURCE_PATH" "$REPO_DIR"
echo $PWD
# Navigate to the repository
cd "$REPO_DIR" || { echo "Repository directory not found!"; exit 1; }

# Add changes to git
git add .

# Commit with a message
COMMIT_MSG="Added/Updated: $BASENAME"
git commit -m "$COMMIT_MSG"

# Push to the remote repository
git push

echo "Successfully pushed $BASENAME to the repository."
