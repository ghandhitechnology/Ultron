#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root inside the guest." >&2
  exit 1
fi

install -m 0755 ultron_guest_agent.py /usr/local/sbin/ultron-guest-agent
cat >/etc/systemd/system/ultron-guest-agent.service <<'UNIT'
[Unit]
Description=Ultron guest probe service
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /usr/local/sbin/ultron-guest-agent
Restart=on-failure
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable ultron-guest-agent.service
