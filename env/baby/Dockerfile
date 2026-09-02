FROM alpine:3.20
RUN apk add --no-cache python3 \
    && adduser -D -u 1000 attacker \
    && mkdir -p /etc/ultron \
    && printf '%s\n' '{"profile_id":"web","misconfig_ids":[]}' >/etc/ultron/profile.json
COPY serve.py /opt/ultron-baby/serve.py
EXPOSE 8080
CMD ["python3", "/opt/ultron-baby/serve.py"]
