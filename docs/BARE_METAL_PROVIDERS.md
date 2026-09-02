# Bare-metal GPU rentals for Ultron

Date: 2026-09-02

Ultron needs a single-tenant x86-64 host with native `/dev/kvm` (not nested), Ubuntu 22.04 or 24.04, two or more NVIDIA 80 GB GPUs, ≥32 physical cores, ≥128 GB RAM, ≥1 TB local NVMe, root, and permission to run libvirt plus an isolated guest bridge. See [SERVER_GUIDE.md](SERVER_GUIDE.md) and `configs/bm-gpu.yaml`.

## Conclusion

Rent **Oracle Cloud `BM.GPU.H100.8`** (or `BM.GPU.A100-v2.8`) first. OCI documents those shapes as bare metal with no hypervisor, publishes 8×80 GB GPUs, 112–128 OCPU, 2 TiB RAM, tens of TB NVMe, Ubuntu 22.04/24.04 images, and ships `oci-kvm` for libvirt guests on the host. List price cited by Oracle for H100 is $10/GPU-hour ($80/node-hour); quota is the usual blocker.

If Oracle quota fails, use **Voltage Park on-demand HGX H100**: they sell self-serve bare-metal H100 80 GB SXM5 from $1.99/GPU-hour, Ubuntu 22.04, SSH as `ubuntu`. KVM is not documented; treat the first hour as an M0 gate.

Third: **Lambda Private Cloud** if you can wait for a quote. Compute nodes are documented as single-tenant bare metal with Ubuntu LTS and root after handoff. Lambda On-Demand Cloud is VMs and is not a substitute. Lambda 1-Click Clusters are single-tenant GPU hardware but start at 16 GPUs and do not document native KVM.

## Comparison

| Provider | SKU | GPUs | CPU | RAM | Disk | KVM | OS | Price | Link |
|---|---|---|---|---|---|---|---|---|---|
| Oracle | `BM.GPU.H100.8` | 8× H100 80 GB | 112 OCPU | 2048 GB | 16× 3.84 TB NVMe | Native BM; Oracle documents KVM on BM, nested KVM unsupported | Ubuntu 22.04/24.04 images; sudo | $10/GPU-hr (Oracle blog); official list is GPU-hr with no public $ | [shapes](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm) |
| Oracle | `BM.GPU.A100-v2.8` | 8× A100 80 GB | 128 OCPU | 2048 GB | 27.2 TB NVMe (4 drives) | Same as above | Same | Official list: GPU-hr, $ unknown | [shapes](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm) |
| Voltage Park | On-demand HGX H100 | 8× H100 80 GB SXM5 per HGX node | 2× Xeon 8480C; core count unpublished | unpublished | OS 2× 1.92 TB NVMe RAID1; data 8× 3.84 TB NVMe RAID0 | Claims no hypervisor; `/dev/kvm` unknown | Ubuntu 22.04 only | From $1.99/GPU-hr | [pricing](https://voltagepark.com/pricing) |
| Lambda | Private Cloud compute node | Cluster SKU unpublished; H100/B200 family | unpublished | unpublished | unpublished | BM claimed; KVM unknown | Ubuntu LTS, root after handoff | Quote | [security posture](https://docs.lambda.ai/private-cloud/security-posture/) |
| Lambda | 1CC GPU node | 16–512 H100 or B200 SXM (cluster min 16) | unpublished per node | unpublished | 24 TB usable NVMe/node | Single-tenant HW; KVM unknown | Ubuntu 22.04 + Lambda Stack; reconfigure allowed | H100 $5.54–$6.16/GPU-hr (2 wk–1 yr) | [1CC](https://docs.lambda.ai/public-cloud/1-click-clusters/) |
| Latitude.sh | `g3.h100.small` | **1×** H100 80 GB | unpublished | unpublished | 2× 3.8 TB NVMe | BM product; KVM unknown | Ubuntu / iPXE / IPMI ISO | $1.68/hr, $1230/mo | [pricing](https://www.latitude.sh/pricing) |
| Latitude.sh | Custom Metal GPU | Wishlist: ask for 2+× H100 | unknown | unknown | unknown | Same | Same | Quote | [wishlist](https://www.latitude.sh/metal-gpu-wishlist) |
| Fluidstack | Dedicated BM cluster | From 8 GPUs (docs: labs/enterprise) | unpublished | unpublished | unpublished | BM under managed K8s/Slurm; host KVM unknown | Atlas-managed images | Quote | [docs](https://docs.fluidstack.io/) |
| RunPod | “Bare metal” reservation | unpublished SKU | unpublished | unpublished | unpublished | unpublished | unpublished | Talk to sales | [release notes](https://docs.runpod.io/release-notes) |
| IBM Classic BM | Fast/custom BM + NVIDIA add-on | GPU model unpublished | Configurable; Ubuntu 22.04 listed for SPR | Configurable | Configurable | No hypervisor on classic BM | Ubuntu 22.04 among options; no-OS | Hourly/monthly; GPU SKU $ unknown | [about BM](https://cloud.ibm.com/docs/bare-metal?topic=bare-metal-about-bm) |

KVM status: **native** = vendor says no customer hypervisor / documents host KVM. **unknown** = searched vendor docs, no `/dev/kvm` or nested-virt statement.

## Per-provider notes

### Oracle Cloud bare metal (best documented fit)

[Compute shapes](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm) list `BM.GPU.H100.8` (8× H100, 112 OCPU, 2048 GB, 16× 3.84 TB NVMe, 1×100 Gbps + 8×2×200 Gbps RDMA) and `BM.GPU.A100-v2.8` (8× A100, 128 OCPU, 2048 GB, 27.2 TB NVMe). [Oracle’s H200 GA post](https://blogs.oracle.com/cloud-infrastructure/now-ga-largest-ai-supercomputer-oci-nvidia-h200) calls OCI GPU compute “bare metal (no hypervisor)” and lists H100 at **$10 per GPU/hour**. The [public price list](https://www.oracle.com/cloud/compute/pricing/) bills those shapes per GPU-hour but the scrape of that page has empty dollar cells.

[Platform images](https://docs.oracle.com/en-us/iaas/Content/Compute/References/images.htm) include Ubuntu 22.04 LTS and 24.04 LTS; `ubuntu` has sudo. [oci-kvm](https://docs.oracle.com/en-us/iaas/oracle-linux/oci/oci-kvm-utility.htm) creates libvirt guests on OCI instances. Oracle Linux KVM docs: KVM on a **bare metal host**; “nested virtualization scenarios aren’t supported for KVM deployments” ([OL10 KVM guide](https://docs.oracle.com/en/operating-systems/oracle-linux/10/kvm-user/OL10-KVM-USER.pdf)).

Firewall: OCI VCNs and security lists; you can add nftables on the host. Isolated libvirt bridge: use a host-only libvirt net (do not attach guest NICs to the VCN). Regions: GPU shapes are not in every AD; check the console.

Deal-breakers: service-limit / quota. Windows images unsupported on these GPU BM shapes (irrelevant).

### Voltage Park / Lightning AI

[Bare metal access](https://www.voltagepark.com/bare-metal-access) and [dedicated reserve](https://www.voltagepark.com/product/dedicated-reserve-compute): no shared hypervisor, HGX H100 80 GB SXM5, 3200 Gbps IB on IB SKUs. [Enterprise specs](https://voltagepark.com/enterprise): 640 GB HBM3 total (8×80), 2× Intel Xeon 8480C, OS 2×1.92 TB NVMe RAID1, data 8×3.84 TB NVMe RAID0. Physical core count and host RAM: **unpublished**.

[Pricing](https://voltagepark.com/pricing): on-demand Ethernet 1–1016 HGX H100; IB 8–1016; reserve 32+ GPUs / 6+ months. FAQ: one H100 from **$1.99/hour**, no contract. [Support FAQ](https://support.voltagepark.com/article/fa-qs): Ubuntu Server 22.04 LTS only, SSH `ubuntu`, no other OS, no GUI. Persistent storage off by default.

KVM: **unknown**. If the node is truly BM, M0 should pass; verify before a long rent. Regions: “six” US DCs on marketing pages; exact metros unpublished on the pages cited.

Deal-breakers: locked image (no 24.04 / custom ISO stated); KVM undocumented; dedicated reserve minimums (32–64 GPUs) are oversized for one researcher.

### Lambda

**On-Demand Cloud:** [docs](https://docs.lambda.ai/public-cloud/on-demand/) — “GPU-backed **virtual machine** instances.” 2× H100 SXM: 52 vCPU, 450 GiB, 5.5 TiB, 80 GB/GPU. Images: Ubuntu 22.04/24.04 Server or Lambda Stack. Firewall: inbound rules, SSH/22 default. [Pricing](https://lambda.ai/pricing): 2× H100 SXM $4.19/GPU-hr. [GH200 troubleshooting](https://docs.lambda.ai/public-cloud/on-demand/troubleshooting/) states GH200s are virtualized. Nested KVM: **unpublished**. **Do not use for Ultron** unless M0 proves native KVM (unlikely on a documented VM).

**1-Click Clusters:** [intro](https://docs.lambda.ai/public-cloud/1-click-clusters/) — 16–512 H100 or B200, Ubuntu 22.04 + Lambda Stack, 24 TB NVMe/compute node, SSH via head nodes, customer firewall on heads. [Security](https://docs.lambda.ai/public-cloud/1-click-clusters/security-posture/): compute nodes are **single-tenant hardware**; heads are multi-tenant VMs. KVM on compute: **unknown**. Price: H100 $6.16 / $5.85 / $5.54 per GPU-hr for 16 / 64 / 256 GPU, 2 weeks–1 year. Minimum 16 GPUs is far above Ultron’s dual-GPU need.

**Bare Metal Instances (2026):** [blog 2026-05-21](https://lambda.ai/blog/lambda-bare-metal-instances) — no third-party hypervisor, SKU example `gpu_metal-4x_gb300` on GB300 NVL72 / Vera Rubin Superclusters. Not a self-serve 2× H100 box.

**Private Cloud:** [security posture](https://docs.lambda.ai/private-cloud/security-posture/) — all compute/head nodes “single-tenant bare-metal”, Ubuntu LTS, customer SSH key, root, dedicated DIA + customer firewall (default no ingress). Managed K8s option **removes SSH** to nodes. Quote-only.

Regions (ODC): US, EU, Japan, India, Israel — [ODC overview](https://docs.lambda.ai/public-cloud/on-demand/).

### Latitude.sh

[Pricing](https://www.latitude.sh/pricing): Metal GPU `g3.h100.small` = **1× H100 80 GB**, 2× 3.8 TB NVMe, $1.68/hr or $1230/mo. CPU/RAM **unpublished**. Also 8× B300 and 8× RTX PRO 6000 (96 GB), not A100/H100 80 GB pairs. [OS](https://www.latitude.sh/docs/servers/operating-systems): catalog + Ubuntu “ML-in-a-Box”; [iPXE](https://www.latitude.sh/docs/servers/custom-images) and [IPMI ISO](https://www.latitude.sh/docs/guides/ipmi-custom-os-deployment) for custom Ubuntu. [Deploy](https://www.latitude.sh/docs/servers/deploying-a-server): custom build / waitlist if stock missing. KVM: **unknown** (product is metal). [Wishlist](https://www.latitude.sh/metal-gpu-wishlist) for other GPU counts.

Deal-breaker on the published H100 SKU: one GPU. Two `g3.h100.small` nodes are two hosts, not Ultron’s single trainer.

### Fluidstack

[Docs](https://docs.fluidstack.io/): managed Kubernetes and Slurm on “bare metal”. No public dual-H100 SKU, CPU/RAM/disk, Ubuntu reimage, or KVM policy. Quote / lab-scale. Fine if they hand you a raw Ubuntu node; unknown until the contract says so.

### RunPod

[Overview](https://docs.runpod.io/overview): Pods are **containerized** GPU/CPU instances. [Instant Clusters](https://docs.runpod.io/instant-clusters) and [blog](https://www.runpod.io/blog/instant-clusters-runpod): Docker, minutes, not full system access. [Release notes](https://docs.runpod.io/release-notes) mention “Bare metal: Reserve dedicated GPU servers” with no SKU, OS, or KVM page found. [Older product post](https://runpod.ghost.io/bare-metal-vs-instant-clusters-for-ai-workload/): Bare Metal = full physical server, monthly/committed, H100/A100, “full” system access. Treat as sales-only; do not buy Pods.

### IBM Cloud

[Classic BM](https://cloud.ibm.com/docs/bare-metal?topic=bare-metal-about-bm): single-tenant, “provisioned without a hypervisor”, hourly/monthly, Ubuntu 22.04 on Sapphire Rapids, optional “No OS”, hardware firewall add-on. NVIDIA: “for certain servers… look for GPU in the Features column.” **GPU model unpublished** on that page — confirm A100/H100 80 GB in the order UI or skip.

[VPC GPU profiles](https://cloud.ibm.com/docs/vpc?topic=vpc-profiles): `gx3d-48x240x2a100p` (2× A100 80 GB) and `gx3d-160x1792x8h100` are **virtual server** profiles. [Release notes](https://cloud.ibm.com/docs/vpc?topic=vpc-release-notes) call H100 a VSI on a sole-tenant HGX host. Nested KVM: **unknown**. Do not assume native KVM.

### Crusoe (not recommended)

[VM overview](https://docs.crusoecloud.com/compute/virtual-machines/overview.md): GPU **VMs**, including `a100-80gb.2x` (2× A100 80 GB PCIe, 24 vCPU, 240 GB RAM, 2×960 GB NVMe) and `h100-80gb-sxm-ib.8x`. Images `ubuntu22.04`. [GPU troubleshooting](https://docs.crusoecloud.com/resources/troubleshooting.md): “full hardware passthrough … into the Virtual Machines.” That is a guest with GPU PT, not a BM hypervisor host. Nested KVM: **unknown**. Skip unless M0 is proven.

### CoreWeave (usually not)

[H100 IB instance](https://docs.coreweave.com/platform/instances/gpu/gd-8xh100ib-i128): `gd-8xh100ib-i128`, 8× H100 80 GB, 128 vCPU, 2048 GB, 61.44 TB. [Security architecture](https://docs.coreweave.com/security/architecture): CKS nodes are bare-metal Kubernetes with BlueField, “without relying on hypervisors.” [Virtual Servers](https://v1.docs.coreweave.com/virtual-servers/deployment-methods/kubectl) are KubeVirt VMs. Researcher path is typically a VM or a K8s node, not a libvirt Ubuntu host. KVM/libvirt on CKS: **unknown**. Quote/enterprise.

## Do not use

| Product | Why | Cite |
|---|---|---|
| Lambda On-Demand | Documented VMs | [ODC overview](https://docs.lambda.ai/public-cloud/on-demand/) |
| RunPod Pods | Containers / templates | [overview](https://docs.runpod.io/overview), [get started](https://docs.runpod.io/get-started) |
| RunPod Instant Clusters | Docker multi-node | [clusters](https://docs.runpod.io/instant-clusters), [launch blog](https://www.runpod.io/blog/instant-clusters-runpod) |
| Vast.ai Docker | Container rental on someone else’s host | [VM vs Docker](https://docs.vast.ai/guides/instances/virtual-machines) |
| Vast.ai “VM” | You are a KVM **guest** (`vastai/kvm` images). Ultron guests would be nested. Hosts enable KVM for **renters**, not so renters become hypervisors | [renter VMs](https://docs.vast.ai/guides/instances/virtual-machines), [host VMs](https://docs.vast.ai/host/vms) |
| TensorDock | “KVM Virtualization” marketplace VMs | [tensordock.com](https://www.tensordock.com/) |
| Paperspace / DigitalOcean GPU Droplets | Managed GPU VMs (not in the BM table; no first-party BM 80 GB SKU found in this pass) | — |
| OVH Public Cloud GPU | “virtualised by the KVM hypervisor”, PCI passthrough | [H100 instances](https://www.ovhcloud.com/en/public-cloud/gpu/h100/), [deploy guide](https://docs.ovhcloud.com/en/guides/public-cloud/compute/deploy-a-gpu-instance) |
| OVH Bare Metal Scale-GPU / HGR-AI | Published GPUs are L4 / L40S, not 80 GB H100/A100 | [AI servers](https://www.ovhcloud.com/en-gb/bare-metal/ai-server/) |
| Hetzner GEX | RTX workstation GPUs; FAQ: no H100 | [GPU matrix](https://www.hetzner.com/dedicated-rootserver/matrix-gpu) |
| phoenixNAP BMC GPU | Intel Max 1100, not NVIDIA 80 GB | [GPU servers](https://phoenixnap.com/bare-metal-cloud/gpu-servers) |
| Equinix Metal | Sunset 2026-06-30 | [Metal docs](https://docs.equinix.com/metal/) |
| AWS EC2 P5 / “Dedicated” GPU | P5 is Nitro (no metal SKU). Nested virt supported only on listed C/M/R/I types, not P5 | [nested virt](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/amazon-ec2-nested-virtualization.html), [accel specs](https://docs.aws.amazon.com/ec2/instance-types/ac.html) |
| IBM VPC `gx3d-*` GPU | Virtual server instances | [profiles](https://cloud.ibm.com/docs/vpc?topic=vpc-profiles) |

## Verify after renting

Follow [SERVER_GUIDE.md](SERVER_GUIDE.md) §1–4 and M0. First paid hour:

```bash
lscpu | egrep 'Model name|Socket|Core|Virtualization'
test -c /dev/kvm && ls -l /dev/kvm
kvm-ok
nvidia-smi --query-gpu=index,name,memory.total --format=csv
./scripts/bootstrap_bm.sh
# throwaway Ubuntu 18.04 guest boot (SERVER_GUIDE M0)
```

`kvm-ok` must say acceleration can be used. `/dev/kvm` alone is not enough. If the provider only offers nested virt, leave.

## Sources

- Ultron: [docs/SERVER_GUIDE.md](SERVER_GUIDE.md), `configs/bm-gpu.yaml`
- Oracle: [shapes](https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm), [images](https://docs.oracle.com/en-us/iaas/Content/Compute/References/images.htm), [oci-kvm](https://docs.oracle.com/en-us/iaas/oracle-linux/oci/oci-kvm-utility.htm), [OL10 KVM PDF](https://docs.oracle.com/en/operating-systems/oracle-linux/10/kvm-user/OL10-KVM-USER.pdf), [H200/H100 price blog](https://blogs.oracle.com/cloud-infrastructure/now-ga-largest-ai-supercomputer-oci-nvidia-h200), [compute pricing](https://www.oracle.com/cloud/compute/pricing/)
- Voltage Park: [bare metal](https://www.voltagepark.com/bare-metal-access), [pricing](https://voltagepark.com/pricing), [enterprise H100 spec](https://voltagepark.com/enterprise), [FAQ OS](https://support.voltagepark.com/article/fa-qs), [dedicated reserve](https://www.voltagepark.com/product/dedicated-reserve-compute)
- Lambda: [ODC](https://docs.lambda.ai/public-cloud/on-demand/), [ODC troubleshooting](https://docs.lambda.ai/public-cloud/on-demand/troubleshooting/), [pricing](https://lambda.ai/pricing), [1CC](https://docs.lambda.ai/public-cloud/1-click-clusters/), [1CC security](https://docs.lambda.ai/public-cloud/1-click-clusters/security-posture/), [Private Cloud](https://docs.lambda.ai/private-cloud/security-posture/), [BM instances blog](https://lambda.ai/blog/lambda-bare-metal-instances)
- Latitude.sh: [pricing](https://www.latitude.sh/pricing), [OS](https://www.latitude.sh/docs/servers/operating-systems), [custom images](https://www.latitude.sh/docs/servers/custom-images), [IPMI ISO](https://www.latitude.sh/docs/guides/ipmi-custom-os-deployment), [wishlist](https://www.latitude.sh/metal-gpu-wishlist)
- Fluidstack: [docs](https://docs.fluidstack.io/)
- RunPod: [overview](https://docs.runpod.io/overview), [clusters](https://docs.runpod.io/instant-clusters), [release notes](https://docs.runpod.io/release-notes), [clusters blog](https://www.runpod.io/blog/instant-clusters-runpod), [BM vs clusters](https://runpod.ghost.io/bare-metal-vs-instant-clusters-for-ai-workload/)
- Vast.ai: [renter VMs](https://docs.vast.ai/guides/instances/virtual-machines), [host VMs](https://docs.vast.ai/host/vms), [verification](https://docs.vast.ai/host/understanding-verification)
- Crusoe: [VM specs](https://docs.crusoecloud.com/compute/virtual-machines/overview.md), [GPU passthrough](https://docs.crusoecloud.com/resources/troubleshooting.md)
- CoreWeave: [H100 IB](https://docs.coreweave.com/platform/instances/gpu/gd-8xh100ib-i128), [architecture](https://docs.coreweave.com/security/architecture)
- IBM: [classic BM](https://cloud.ibm.com/docs/bare-metal?topic=bare-metal-about-bm), [VPC profiles](https://cloud.ibm.com/docs/vpc?topic=vpc-profiles)
- OVH: [H100 public cloud](https://www.ovhcloud.com/en/public-cloud/gpu/h100/), [deploy GPU](https://docs.ovhcloud.com/en/guides/public-cloud/compute/deploy-a-gpu-instance), [AI BM](https://www.ovhcloud.com/en-gb/bare-metal/ai-server/)
- Hetzner: [GPU matrix](https://www.hetzner.com/dedicated-rootserver/matrix-gpu)
- phoenixNAP: [GPU servers](https://phoenixnap.com/bare-metal-cloud/gpu-servers)
- Equinix: [Metal EOL](https://docs.equinix.com/metal/)
- AWS: [nested virt](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/amazon-ec2-nested-virtualization.html), [accelerated types](https://docs.aws.amazon.com/ec2/instance-types/ac.html)
- TensorDock: [home](https://www.tensordock.com/)
