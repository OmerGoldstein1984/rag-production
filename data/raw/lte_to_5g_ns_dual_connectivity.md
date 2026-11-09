# LTE-to-5G Non-Standalone Dual Connectivity

## Effective Date: January 1, 2023
## Version: 1.0
## Topic: LTE / 5G
## Source: 3GPP TS 37.340, TS 37.335, TS 36.331, TS 38.331
## Access Level: public

### Overview

Non-Standalone (NSA) 5G, defined as 3GPP Option 3x, was the first 5G deployment mode. In NSA, an LTE network (E-UTRAN + EPC) acts as the anchor and a 5G NR (gNB) provides additional capacity via dual connectivity. The UE is simultaneously connected to an LTE eNodeB (master) and a 5G gNB (secondary), aggregating bandwidth from both radio technologies.

### Architecture (EN-DC, Option 3x)

- **EN-DC (E-UTRA-NR Dual Connectivity):** The Option 3x architecture where an eNodeB (MeNB) and a gNB (SgNB) serve the same UE. The EPC (MME/S-GW/P-GW) remains the core network anchor — there is no 5GC in pure Option 3x.
- **MeNB (Master eNodeB):** The LTE eNodeB that handles RRC connection to the UE, mobility management, and carries the MCG (Master Cell Group). The MeNB's S1 connection to the MME anchors the UE.
- **SgNB (Secondary gNodeB):** The 5G NR base station providing additional radio resources (SCG — Secondary Cell Group). Connected to the MeNB via the X2-C interface (eNodeB-gNodeB, using X2AP extended for NR).

### Bearer Types in EN-DC

- **MCG bearer:** Carried only over LTE (eNodeB to S-GW via S1-U). Includes the SRB1 (signaling) and default data bearer.
- **SCG bearer:** Carried only over NR (gNB to... via X2-C to MeNB, then S1-U to S-GW). Used for additional data capacity.
- **Split bearer:** Carried over both LTE and NR, with PDCP at the MeNB duplicating/splitting packets to both eNodeB and gNB. Used for reliability or to balance load across radios.

### Interfaces

- **X2-C (extended):** MeNB to SgNB (control plane) for SgNB addition/modification/release, handover coordination, and UL/DL path switching.
- **S1-U:** MeNB (eNodeB) to S-GW (user plane) for all bearers' user-plane anchoring — even SCG traffic ultimately egresses via the MeNB/S1-U in Option 3x.
- **Uu (LTE):** UE to MeNB eNodeB.
- **Uu (NR):** UE to SgNB gNB.

### Procedures

- **NSA attach:** The UE first camps on an LTE cell and performs a normal LTE attach via the MeNB/MME. After the LTE RRC connection is established, the MeNB configures an NR SgNB via X2-C (SgNB addition), and the UE establishes an NR RRC connection to the gNB while keeping the LTE RRC connection.
- **Carrier aggregation:** The MeNB can aggregate LTE carriers (LTE CA) while simultaneously using the NR SCG, giving the UE combined bandwidth from LTE + NR.
- **Mobility:** LTE mobility (handover to another eNodeB) is controlled by the MeNB/MME. NR mobility (adding/releasing an SgNB or changing NR cells) can occur without changing the LTE anchor. The UE can move between EN-DC and standalone LTE (SgNB release) without dropping the call.
- **Fallback to 5G SA:** When the network deploys a 5GC (AMF/SMF/UPF), the same gNB can switch from Option 3x (NSA, eNodeB anchor) to Option 2/5 (SA, gNB anchor) without changing the radio.

### Abbreviations

- NSA: Non-Standalone
- EN-DC: E-UTRA-NR Dual Connectivity
- MeNB: Master eNodeB
- SgNB: Secondary gNodeB
- MCG: Master Cell Group
- SCG: Secondary Cell Group
- S1-U: S1 User Plane
- X2-C: X2 Control Plane
- SRB: Signaling Radio Bearer
- PDCP: Packet Data Convergence Protocol
