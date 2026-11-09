# eSIM Remote Provisioning

## Effective Date: January 1, 2023
## Version: 1.0
## Topic: eSIM
## Source: GSMA SGP.02, SGP.21, SGP.22, SGP.31
## Access Level: public

### Overview

eSIM (embedded SIM) replaces the physical removable SIM card with a soldered eUICC (embedded Universal Integrated Circuit Card) that can hold multiple operator profiles. Remote provisioning allows operator profiles to be downloaded, installed, enabled, disabled, and deleted over the air without physically swapping a card. This is managed by the GSMA remote provisioning architecture for embedded UICC (M2M) and consumer devices.

### Key Entities

- **eUICC (embedded UICC):** The secure element embedded in the device. Runs the ISD-R (Issuer Security Domain - Root), ISD-P (Profile), and ECASD (eUICC Certificate Authority Security Domain). Stores one or more operator profiles.
- **SM-DP+ (Subscription Manager Data Preparation+):** Prepares operator profiles (binds them to a specific eUICC via EID), signs them, and delivers them to the eUICC over the ES9+ interface. Operates on behalf of the mobile network operator (MNO).
- **SM-SR (Subscription Manager Secure Routing):** Manages eUICC lifecycle and secure channel (ES8+/ES12). Handles profile installation, enabling, disabling, and deletion, and can authorize profile changes by another operator.
- **LPA (Local Profile Assistant):** Software on the device (e.g., in the OS or eUICC API layer) that discovers SM-DP+ services, initiates profile download, and manages the local profile UI (settings). LPA can be embedded (eLPA) or generic (e.g., a downloadable app).
- **EID (eUICC Identifier):** Unique identifier of the eUICC, used to bind a profile to a specific device during provisioning.

### Remote Provisioning Workflow

1. **Discovery:** The LPA obtains the SM-DP+ address (e.g., via a QR code containing an activation code: SM-DP+ URL + matching code).
2. **Profile preparation:** The operator provisions a profile on the SM-DP+, bound to the device's EID. The SM-DP+ generates and signs the profile package.
3. **Download and installation:** The LPA initiates a secure session (ES9+) to the SM-DP+, authenticates using the matching code and eUICC certificates, downloads the profile, and installs it as an ISD-P on the eUICC.
4. **Enable/Disable:** The installed profile is initially disabled. The LPA (via SM-SR ES12) sends an enable command, making the profile active for network access. Disabling retains the profile but takes it offline.
5. **Deletion:** A profile can be deleted from the eUICC (typically authorized by the original SM-DP+ or SM-SR).

### Interfaces (GSMA References)

- **ES8+:** eUICC to SM-SR (eUICC lifecycle management).
- **ES9+:** LPA/eUICC to SM-DP+ (profile download and installation).
- **ES12:** SM-SR to SM-SR / LPA to SM-SR (profile enable/disable/delete authorization).
- **ES15:** SM-DP+ to SM-DP+ (profile transfer between operators, e.g., for profile ownership change).

### Consumer vs. M2M

- **Consumer eSIM (SGP.21/SGP.22):** User-initiated profile download via QR code or app. LPA runs on a smartphone OS (iOS/Android). Supports multiple profiles, one enabled at a time.
- **M2M eSIM (SGP.02/SGP.31):** Machine-to-machine, often with minimal user interaction. SM-SR plays a stronger role in remote management. Used in IoT, automotive, and enterprise devices.

### Abbreviations

- eSIM: Embedded SIM
- eUICC: Embedded Universal Integrated Circuit Card
- SM-DP+: Subscription Manager Data Preparation+
- SM-SR: Subscription Manager Secure Routing
- LPA: Local Profile Assistant
- EID: eUICC Identifier
- ISD-P: Issuer Security Domain - Profile
- ISD-R: Issuer Security Domain - Root
- ECASD: eUICC Certificate Authority Security Domain
