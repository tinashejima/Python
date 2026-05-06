#!/bin/bash

# Load environment variables
source .env

# Define directories
ROOT_DIR="/home/takue/Documents/Projects/Zim-ttech/deployment"
CLIENT_DIR="$ROOT_DIR/client"
# CLIENT_LIBS_DIR="$ROOT_DIR/client-libs"
SERVER_DIR="$ROOT_DIR"

# Checkout repositories
# echo "Checking out server repository..."
# git clone -b "$SERVER_BRANCH" https://github.com/mohcc/server.git "$SERVER_DIR"

# echo "Checking out client repository..."
# git clone -b "$CLIENT_BRANCH" https://github.com/mohcc/client.git "$CLIENT_DIR"

# echo "Checking out client-libs repository..."
# git clone https://github.com/mohcc/client-libs.git "$CLIENT_LIBS_DIR"

# Install Node.js and Yarn
# echo "Setting up Node.js..."
# export NVM_DIR="$HOME/.nvm"
# . "$NVM_DIR/nvm.sh"
# nvm install 18.12.0
# nvm use 18.12.0

# echo "Installing Yarn..."
# npm install -g yarn

# Setup client libraries
# echo "Setting up client libraries..."
# mkdir -p "$CLIENT_DIR/libraries"
# unzip "$CLIENT_LIBS_DIR/lib1.zip" -d "$CLIENT_DIR/libraries"
# unzip "$CLIENT_LIBS_DIR/lib2.zip" -d "$CLIENT_DIR/libraries"

# Install dependencies
# echo "Installing Nx CLI..."
# yarn global add nx

# echo "Installing client dependencies..."
 cd "$CLIENT_DIR" || exit
# yarn install

# Build the client application

echo "Building client application..."
export NODE_OPTIONS="--max_old_space_size=4096"
nx build ehr-web --prod

# Copy build files to server
echo "Copying built files to server directory..."
cp -r ./dist/apps/ehr-web/* "$SERVER_DIR/mrs-web/src/main/webapp/"

# Setup Java
# echo "Setting up Java..."
# export JAVA_HOME=$(/usr/libexec/java_home -v "$JAVA_VERSION")
# export PATH=$JAVA_HOME/bin:$PATH

# Build server with Maven
echo "Building server..."
cd "$SERVER_DIR" || exit
mvn clean -U install -DskipTests
cd mrs-web || exit
mvn -U verify -DdockerImageRepo="$DOCKER_IMAGE_REPO" -DdockerImageTag="$DOCKER_IMAGE_TAG" -Pprod dockerfile:build

# Docker login
echo "Logging in to Docker..."
echo "$DOCKERHUB_PASSWORD" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

# Push Docker image
echo "Pushing Docker image..."
docker push "$DOCKER_IMAGE_REPO:$DOCKER_IMAGE_TAG"

# SSH into server and deploy
echo "Deploying to server..."
sshpass -p "$SSH_SERVER_PASSWORD" ssh -o StrictHostKeyChecking=no "$SSH_SERVER_USERNAME@$SSH_SERVER_HOST" -p "$SSH_SERVER_PORT" << EOF
  export VERSION="$DOCKER_IMAGE_TAG"
  cd "$DEPLOYMENT_DIR"
  docker-compose down
  docker rmi -f "$DOCKER_IMAGE_REPO:$DOCKER_IMAGE_TAG"
  docker-compose up -d
EOF

echo "Deployment complete!"
