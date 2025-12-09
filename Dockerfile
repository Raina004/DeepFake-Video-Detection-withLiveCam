FROM node:18-alpine

# Install Python and required system dependencies
RUN apk add --no-cache python3 py3-pip ffmpeg

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci

# Copy Python requirements and install
COPY scripts/requirements.txt ./scripts/
RUN pip3 install -r scripts/requirements.txt

# Copy application code
COPY . .

# Build Next.js application
RUN npm run build

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
