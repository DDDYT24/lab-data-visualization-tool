FROM postgres:17.11-alpine3.24

RUN apk add --no-cache aws-cli python3

COPY deploy/aws/backup/logical_backup.py /opt/labviz/logical_backup.py

USER postgres
ENTRYPOINT ["python3", "/opt/labviz/logical_backup.py"]
