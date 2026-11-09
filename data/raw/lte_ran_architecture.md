# LTE RAN Architecture

## Effective Date: January 1, 2023
## Version: 1.0
## Topic: LTE
## Source: 3GPP TS 36.300, TS 36.413, TS 36.423
## Access Level: public

### Overview

LTE (Long-Term Evolution) is the 4G mobile radio access technology specified by 3GPP. The LTE radio access network is called E-UTRAN (E-UMTS Terrestrial Radio Access Network). It is a flat, all-IP architecture with a single node type — the eNodeB (evolved Node B) — that handles radio resource management, scheduling, and ciphering.

### Key Network Elements

- **eNodeB (evolved Node B):** The LTE base station. Terminates the Uu air interface toward the UE (User Equipment) and the S1/X2 interfaces toward the core network and neighboring eNodeBs. Each eNodeB controls one or more cells (sectors), identified by a Physical Cell ID (PCI) and E-ARFCN (E-UTRA Absolute Radio Frequency Channel Number).
- **MME (Mobility Management Entity):** Control-plane node in the EPC (Evolved Packet Core). Handles bearer management, authentication, paging, and tracking area management.
- **S-GW (Serving Gateway):** User-plane anchor for the UE within the EPC. Routes data between eNodeBs and the P-GW as the UE moves.
- **P-GW (PDN Gateway):** User-plane anchor to external packet data networks (internet/IMS). Assigns the UE's IP address and enforces policy/charging (PCC).

### Key Interfaces

- **Uu:** Air interface between UE and eNodeB (OFDMA downlink, SC-FDMA uplink).
- **S1-MME:** eNodeB to MME (control plane, S1AP protocol).
- **S1-U:** eNodeB to S-GW (user plane, GTP-U protocol).
- **X2:** eNodeB to neighboring eNodeB (handover coordination, load balancing, X2AP protocol).

### Bearers

An **EPS bearer** is the logical conduit carrying user-plane packets between the UE and the P-GW. Each bearer has a QoS profile (QCI — QoS Class Identifier, GBR/non-GBR bit rates, ARP priority).

- **Default bearer:** Established at initial attach; always-on IP connectivity.
- **Dedicated bearer:** Established for specific services requiring differentiated QoS (e.g., VoLTE voice bearer with QCI 1).

### Procedures

- **Attach:** The UE performs cell selection, random access, RRC connection setup, authentication (AKA via MME/HSS), security mode command, and default bearer establishment.
- **Handover:** X2-based handover between eNodeBs (source/target) with S1 path switch via the MME/S-GW. Intra-LTE handovers preserve the IP address.
- **Carrier Aggregation:** Combines multiple E-UTRA carriers (component carriers) into a wider effective bandwidth to increase peak data rates. One cell is the PCell (primary), others are SCells (secondary).

### Abbreviations

- E-UTRAN: E-UMTS Terrestrial Radio Access Network
- EPC: Evolved Packet Core
- UE: User Equipment
- MME: Mobility Management Entity
- S-GW: Serving Gateway
- P-GW: PDN Gateway
- QCI: QoS Class Identifier
- PCRF: Policy and Charging Rules Function
- PCI: Physical Cell ID
- E-ARFCN: E-UTRA Absolute Radio Frequency Channel Number
