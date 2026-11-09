# International Roaming

## Effective Date: January 1, 2023
## Version: 1.0
## Topic: Roaming
## Source: 3GPP TS 23.271, TS 29.060, GSMA IR.21
## Access Level: public

### Overview

International roaming allows a subscriber (UE) of a home PLMN (Public Land Mobile Network) to use services when visiting a foreign country by attaching to a visited PLMN. Roaming requires commercial roaming agreements between operators and technical interworking between their networks (typically via GRX/IPX backbone networks).

### Key Entities

- **HPLMN (Home PLMN):** The subscriber's home operator network, where the IMSI is provisioned and where authentication/subscription data resides (HSS/UDM).
- **VPLMN (Visited PLMN):** The foreign network the subscriber is currently attached to. Provides radio access and local user-plane anchoring.
- **PLMN ID:** MCC (Mobile Country Code, 3 digits) + MNC (Mobile Network Code, 2-3 digits) uniquely identifies a PLMN globally.

### Roaming Architecture

In roaming, the VPLMN's S-GW/P-GW (or 5G UPF/SMF) typically anchors user-plane traffic, with a backhaul tunnel (GTP over GRX/IPX) to the HPLMN for authentication and charging. Two common traffic steering models:

- **Routed roaming (home-routed):** All user-plane traffic tunnels from VPLMN back to HPLMN P-GW/UPF, then to the internet. The HPLMN sees the subscriber's IP and can apply home charging/policy. Higher latency but consistent service.
- **Local breakout:** User-plane traffic breaks out locally at the VPLMN P-GW/UPF to the local internet. Lower latency but requires local charging mediation and regulatory compliance in the visited country.

### Key Interfaces

- **S7:** MME-to-MME interface (home-visited) for context transfer during roaming attach/handover (S1AP-based).
- **S13:** MME to EIR (Equipment Identity Register) — checks device IMEI status (blacklisted/stolen) during attach, including roaming devices.
- **Gx/Gy:** PCRF/OCS policy and charging interfaces between HPLMN and VPLMN elements.
- **GRX/IPX:** Inter-operator IP backbone carrying GTP tunnels between roaming partners' S-GW/P-GW nodes.

### Roaming Procedures

- **Roaming attach:** The UE attaches to the VPLMN eNodeB/gNB. The VPLMN MME/AMF detects the foreign IMSI (MCC/MNC != visited), queries the HPLMN HSS/UDM via the S7/N11 equivalent (often via a Diameter/Gx roaming exchange), obtains authentication vectors, and completes AKA.
- **IMS roaming:** For VoLTE/VoNR roaming, the UE's P-CSCF is discovered in the VPLMN (or home), and IMS registration traverses the visited network's IBCF (IP Border Control Function). The home S-CSCF handles session control.
- **Roaming charging:** The VPLMN generates CDRs (Call Detail Records) for visited usage and sends them to the HPLMN via TAP (Transferred Account Procedure) files for subscriber billing.

### Abbreviations

- PLMN: Public Land Mobile Network
- HPLMN: Home PLMN
- VPLMN: Visited PLMN
- MCC: Mobile Country Code
- MNC: Mobile Network Code
- IMSI: International Mobile Subscriber Identity
- HSS: Home Subscriber Server
- UDM: Unified Data Management (5G)
- GRX: GPRS Roaming Exchange
- IPX: IP Exchange
- S-GW: Serving Gateway
- P-GW: PDN Gateway
- CDR: Call Detail Record
- TAP: Transferred Account Procedure
