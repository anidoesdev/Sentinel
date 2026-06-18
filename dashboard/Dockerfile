FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Baked into the client bundle at build time — set via docker-compose build arg
ARG NEXT_PUBLIC_API_HOST=localhost
ENV NEXT_PUBLIC_API_HOST=$NEXT_PUBLIC_API_HOST

RUN npm run build

ENV NODE_ENV=production
EXPOSE 3000

CMD ["npm", "start"]
