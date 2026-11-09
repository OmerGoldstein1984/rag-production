# 5G Standalone (SA) Core Architecture

## Effective Date: January 1, 2023
## Version: 1.0
## Topic: 5G
## Source: 3GPP TS 23.501, TS 38.401, TS 29.500
## Access Level: public

### Overview

5G NR (New Radio) Standalone (SA) is the 5G mobile architecture that operates without any dependence on LTE. In SA mode, both the radio access network (NG-RAN) and the core network (5GC) are native 5G. The 5GC uses a Service-Based Architecture (SBA) where network functions expose services over HTTP/2 interfaces, enabling flexible composition and cloud-native deployment.

### Key Network Elements (NG-RAN)

- **gNodeB (gNB):** The 5G NR base station. Terminates the Uu air interface toward the UE and the NG interfaces toward the 5GC. A gNB can be a Central Unit (CU) or Distributed Unit (DU) in a split architecture (Option 7 for F1 split).
- **ng-eNB:** An LTE eNodeB upgraded to connect to the 5GC via the NG interface, enabling interworking between LTE and 5GC (used in Option 4/7 migration paths).

### Key Network Elements (5GC)

- **AMF (Access and Mobility Management Function):** The single point of contact for the NG-RAN. Handles registration, access authentication (via AUSF), security, mobility (handovers), and reachability/paging.
- **SMF (Session Management Function):** Manages PDU (Protocol Data Unit) sessions — the 5G equivalent of EPS bearers. Selects the UPF, establishes QoS flows, and anchors user-plane traffic.
- **UPF (User Plane Function):** Anchors user-plane packets. Handles packet routing/forwarding, QoS enforcement (PQI), lawful interception, and traffic steering. Can be deployed at the network edge (MEC) for low latency.
- **AUSF (Authentication Server Function):** Performs AKA-based authentication for 5G.
- **UDM (Unified Data Management):** Stores subscriber profiles and authentication credentials.
- **PCF (Policy Control Function):** Provides policy rules to the SMF and AF (Application Function).
- **NEF (Network Exposure Function):** Exposes 5GC capabilities to external applications securely.

### Key Interfaces (Reference Points)

- **N1:** UE to AMF (non-access stratum signaling, carried over RRC).
- **N2:** gNB to AMF (NGAP protocol, control plane).
- **N3:** gNB to UPF (user plane, GTP-U).
- **N4:** SMF to UPF (PFCP protocol, session management).
- **N6:** UPF to external data network (DN).
- **N11:** AMF to SMF (subscription/mobility events).
- **N32/N33:** Charging (CDR generation) and policy events.

### Service-Based Architecture (SBA)

In the SBA, NFs (AMF, SMF, PCF, UDM, etc.) act as service producers or consumers. The NRF (Network Repository Function) enables service discovery. Interfaces use HTTPS/2 with JSON payloads (OpenAPI), e.g., Nudm, Nsmf, Namf, Nnrf. This contrasts with EPC's point-to-point GTP/Diameter interfaces.

### PDU Session

A **PDU session** is the 5G logical connection between the UE and a data network, established by the SMF and anchored at a selected UPF. Each PDU session carries one or more QoS flows mapped to DRBs (Data Radio Bearers). Session types include IPv4, IPv6, IPv4v6, Ethernet, and Unstructured.

### Abbreviations

- 5GC: 5G Core Network
- NG-RAN: Next Generation Radio Access Network
- gNB: gNodeB (5G base station)
- AMF: Access and Mobility Management Function
- SMF: Session Management Function
- UPF: User Plane Function
- SBA: Service-Based Architecture
- PQI: 5G QoS Identifier
- DRB: Data Radio Bearer
- MEC: Multi-access Edge Computing
