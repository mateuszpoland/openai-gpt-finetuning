FROM python:3.9

WORKDIR /app

ARG GITHUB_ACCESS_TOKEN
ARG OPENAI_API_KEY

ENV GITHUB_ACCESS_TOKEN=${GITHUB_ACCESS_TOKEN}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

COPY . .
COPY requirements.txt .

RUN echo "machine github.com login $GITHUB_ACCESS_TOKEN password x-oauth-basic" > ~/.netrc && \
    chmod 600 ~/.netrc && \
    pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]