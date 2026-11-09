# VoLTE and VoNR Call Flows

## Effective Date: January 1, 2023
## Version: 1.0
## Topic: LTE / 5G
## Source: 3GPP TS 23.228, TS 23.401, TS 23.501, TS 23.203
## Access Level: public

### Overview

VoLTE (Voice over LTE) and VoNR (Voice over New Radio) carry voice as IP packets over the 4G/5G data bearer instead of circuit-switched (CS) voice. Both use the IMS (IP Multimedia Subsystem) for session control (SIP/SDP) and a dedicated QoS bearer for the media. Voice is encoded with the AMR (Adaptive Multi-Rate) codec family.

### IMS Architecture

- **P-CSCF (Proxy-CSCF):** The SIP proxy closest to the UE. First point of contact for IMS registration and session initiation. Discovered via NAS signaling or DHCP.
- **S-CSCF (Serving-CSCF):** The core SIP registrar and session controller for the subscriber's home domain. Handles registration, routing, and service invocation.
- **ICS-CSCF (Interrogating CSCF):** Used in roaming/interconnection to hide the home network topology.
- **HSS:** Stores IMS subscription data and user location (S-CSCF address).
- **MGCF/MGW (Media Gateway Controller/Function):** Interworks IMS (SIP/RTP) with circuit-switched PSTN for calls to non-IMS numbers.

### EPS Bearers for Voice

VoLTE uses a dedicated EPS bearer with **QCI 1** (Guaranteed Bit Rate, for conversational voice). The default bearer (QCI 9, best-effort) carries signaling and background data; the dedicated QCI 1 bearer carries RTP media with prioritized scheduling and low latency.

- **QCI 1:** Conversational voice (GBR, low latency, high priority).
- **QCI 5:** IMS signaling (non-GBR).
- **QCI 9:** Best-effort internet.

### VoLTE Call Flow (simplified)

1. **IMS registration:** UE registers with the P-CSCF/S-CSCF via SIP REGISTER over the default bearer. Authentication via AKA/IMS AKA.
2. **Call setup:** Caller UE sends SIP INVITE to P-CSCF, routed via S-CSCF to callee's S-CSCF. SDP negotiates codecs (e.g., AMR-WB) and RTP ports.
3. **Dedicated bearer setup:** Upon call acceptance, the network establishes a QCI 1 dedicated EPS bearer (EPC dedicated bearer activation, triggered by PCF/PCRF policy).
4. **Media:** RTP/RTCP media flows between UEs over the QCI 1 bearer (P-GW/S-GW anchor, eNodeB scheduling).
5. **Call teardown:** SIP BYE releases the session; the dedicated bearer is released.

### SRVCC (Single Radio Voice Call Continuity)

If a VoLTE call moves out of LTE coverage (e.g., to 2G/3G), SRVCC hands the call over to circuit-switched voice mid-call without dropping. The IMS anchors the call (CS-IEC / MSC server), and the UE switches to CS radio resources while the S-CSCF keeps the SIP session alive.

### VoNR (Voice over New Radio)

VoNR is the 5G SA equivalent of VoLTE. Uses a QCI 1-equivalent 5G QoS flow (5QI 1, conversational voice) over an NR DRB, anchored at the 5G UPF. Session control remains IMS (SIP/SDP). Interworking between VoLTE and VoNR occurs during LTE<->5G mobility (e.g., EN-DC to NR, or NR to LTE fallback).

### Abbreviations

- VoLTE: Voice over LTE
- VoNR: Voice over New Radio
- IMS: IP Multimedia Subsystem
- P-CSCF: Proxy-CSCF
- S-CSCF: Serving-CSCF
- HSS: Home Subscriber Server
- QCI: QoS Class Identifier
- 5QI: 5G QoS Identifier
- RTP: Real-time Transport Protocol
- SDP: Session Description Protocol
- AMR: Adaptive Multi-Rate
- SRVCC: Single Radio Voice Call Continuity
- DRB: Data Radio Bearer
