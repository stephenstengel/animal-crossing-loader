docker build -t "animal-crossing-loader" .
docker run --rm -v %cd%:/app/ -w /app/ animal-crossing-loader python3 load-dataset.py -d
