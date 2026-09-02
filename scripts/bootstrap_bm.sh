#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--install-host-packages" ]]; then
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run --install-host-packages as root." >&2
    exit 2
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cpu-checker qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils \
    cloud-image-utils python3-venv git jq tmux
  systemctl enable --now libvirtd
  echo "Install the recommended NVIDIA datacenter driver, reboot, then rerun without flags."
  exit 0
fi

failures=0
for command in kvm-ok virsh nvidia-smi; do
  if ! command -v "${command}" >/dev/null; then
    echo "MISSING: ${command}" >&2
    failures=$((failures + 1))
  fi
done

if [[ ! -c /dev/kvm ]]; then
  echo "MISSING: native /dev/kvm character device" >&2
  failures=$((failures + 1))
elif ! kvm-ok; then
  failures=$((failures + 1))
fi

if command -v nvidia-smi >/dev/null; then
  nvidia-smi
fi
if command -v virsh >/dev/null && ! virsh -c qemu:///system list >/dev/null; then
  echo "FAILED: libvirt system connection" >&2
  failures=$((failures + 1))
fi
if ! command -v tmux >/dev/null; then
  echo "WARNING: tmux is missing; generation and vLLM jobs will die on SSH disconnect." >&2
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "M0 failed with ${failures} host gate error(s)." >&2
  exit 1
fi

echo "M0 host gates passed. Run a throwaway Ubuntu 18.04 guest before M1."
