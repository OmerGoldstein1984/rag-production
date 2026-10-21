# Acme Corp Product FAQ

## Product: AcmeCloud Storage Service

### General Questions
**Q: What is AcmeCloud Storage Service?**
A: AcmeCloud is a scalable object storage service designed for businesses to store and retrieve any amount of data from anywhere on the web.

**Q: What are the key features of AcmeCloud?**
A: Key features include:
- 99.999999999% (11 nines) durability
- 99.99% availability SLA
- Lifecycle policies for automatic data tiering
- Cross-region replication for disaster recovery
- Fine-grained access control via IAM roles

### Pricing and Billing
**Q: How is AcmeCloud priced?**
A: AcmeCloud pricing is based on:
- Storage usage (per GB per month)
- Number of PUT, COPY, POST, or LIST requests
- Number of GET and other requests
- Data transfer out to the internet
- Data transfer between AcmeCloud regions (if applicable)

**Q: Are there any discounts for long-term commitments?**
A: Yes, Acme Corp offers reserved storage pricing for commitments of 1 or 3 years, providing up to 40% discount over pay-as-you-go rates.

### Technical Questions
**Q: What is the maximum object size I can store in AcmeCloud?**
A: The maximum object size is 5 TB. For objects larger than 5 TB, you must use the multipart upload capability.

**Q: How does AcmeCloud ensure data durability?**
A: AcmeCloud stores data redundantly across multiple devices and facilities within a region. It uses erasure coding and checksums to detect and repair data corruption.

**Q: Can I host a static website on AcmeCloud?**
A: Yes, you can configure an AcmeCloud bucket to host a static website. The service supports index documents and error documents.

### Abbreviations
- SLA: Service Level Agreement
- IAM: Identity and Access Management
- TB: Terabyte
- GB: Gigabyte