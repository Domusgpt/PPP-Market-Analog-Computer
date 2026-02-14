FROM node:20-slim

WORKDIR /app

COPY platform/package.json .
RUN npm install

COPY platform/ .

EXPOSE 5173

CMD ["npx", "vite", "--host", "0.0.0.0", "--port", "5173"]
