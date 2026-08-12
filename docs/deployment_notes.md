# ApneaGuard AI - Deployment Notes

## AWS Budget Alerts & Free-Tier Tracking
**Important Dates:**
- **Today's Date**: August 12, 2026
- **6-Month Free Credit Expiration Date**: February 12, 2027

> [!WARNING]
> **Action Required Before Provisioning:** An AWS Budget Alert MUST be created in the AWS Billing Console, bound to the account email, triggering at a low threshold (e.g., $15.00). Ensure this is active before launching the EC2 instance below to prevent unexpected overage charges.

---

## AWS Architecture & Provisioning (Alternative Local Strategy)
For demonstration and interview purposes, the deployment is orchestrated as a single-node monolithic container stack using `docker-compose.prod.yml`. This prevents any managed-DB cold-start risks or excessive networking costs.

**Target AWS Footprint:**
- **Service**: 1x EC2 Instance (Amazon Linux 2023 or Ubuntu 24.04 LTS).
- **Instance Type**: `t3.micro` (or `t2.micro` depending on regional availability).
- **Storage**: 8 GB EBS General Purpose (gp3) root volume.
- **Estimated On-Demand Cost** *(if free tier lapses in us-east-1)*: 
  - `t3.micro`: ~$7.60 / month.
  - `8GB gp3 EBS`: ~$0.64 / month.
  - **Total**: ~$8.24 / month (excluding data transfer out, which should be minimal).

## Security Group Configuration
To strictly limit the attack surface, the EC2 Security Group must be configured as follows:
- **Inbound Port 80 (HTTP)**: Allow `0.0.0.0/0` (For public web traffic to the dashboard).
- **Inbound Port 443 (HTTPS)**: Allow `0.0.0.0/0` (For SSL termination if configured).
- **Inbound Port 22 (SSH)**: Allow **ONLY YOUR SPECIFIC IP** (e.g., `203.0.113.45/32`). Do not leave this open to `0.0.0.0/0`.
- **Outbound**: Allow all traffic.

---

## Explicit Teardown Procedure

> [!CAUTION]
> This procedure MUST be executed immediately following the completion of the live clinical demo or the ResMed interview process to ensure zero ongoing billing.

Execute these commands via the AWS CLI, or perform the equivalent actions in the AWS Management Console:

1. **Terminate the EC2 Instance**:
   ```bash
   aws ec2 terminate-instances --instance-ids i-0abcd1234efgh5678
   ```

2. **Release any allocated Elastic IP (if used instead of the dynamic public IP)**:
   ```bash
   aws ec2 release-address --allocation-id eipalloc-0123456789abcdef0
   ```

3. **Verify the EBS Volume was deleted** (By default, root volumes are deleted on termination, but always verify):
   ```bash
   aws ec2 describe-volumes --filters Name=attachment.instance-id,Values=i-0abcd1234efgh5678
   # The above should return empty. If a volume exists, delete it:
   aws ec2 delete-volume --volume-id vol-0123456789abcdef0
   ```
