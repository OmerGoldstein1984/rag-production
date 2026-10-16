"""
Golden dataset for evaluating the RAG system on telecom documentation.
Contains question-answer pairs with expected sources and answers.
"""

from typing import List, Dict, Any

# Golden dataset for telecom documentation (eSIM, roaming, LTE/5G, SIM security)
GOLDEN_DATASET = [
    # eSIM Document Questions
    {
        "question": "What is an eSIM and how does it differ from a traditional SIM card?",
        "expected_answer": "An eSIM (embedded SIM) is a programmable SIM card that is embedded directly into a device's hardware, unlike traditional SIM cards which are physical, removable cards. eSIMs can be reprogrammed remotely to switch carriers without physically changing the SIM.",
        "expected_sources": ["eSIM_document.txt"],
        "question_type": "factual"
    },
    {
        "question": "What are the key benefits of eSIM technology for consumers?",
        "expected_answer": "Key benefits of eSIM for consumers include: ability to switch carriers without changing physical SIM cards, support for multiple profiles on a single device, easier international roaming by downloading local carrier profiles, and reduced risk of SIM loss or damage.",
        "expected_sources": ["eSIM_document.txt"],
        "question_type": "benefits"
    },
    {
        "question": "What are the mandatory GSMA requirements for eSIM remote provisioning?",
        "expected_answer": "Mandatory GSMA requirements for eSIM remote provisioning include: secure communication channels using TLS, mutual authentication between devices and SM-DP+ (Subscription Manager Data Preparation), use of ES6 and ES7 interfaces for profile download and management, and compliance with GSMA Remote SIM Provisioning specifications (SGP.01, SGP.02, SGP.11, SGP.21, SGP.22).",
        "expected_sources": ["eSIM_document.txt"],
        "question_type": "requirements"
    },
    {
        "question": "How does the remote provisioning architecture work for eSIM technology?",
        "expected_answer": "The eSIM remote provisioning architecture involves three key components working together securely: Device (eUICC), SM-DP+ (Subscription Manager Data Preparation) responsible for preparing and downloading profiles, and SM-DS+ (Subscription Manager Secure Router) that manages profile metadata and enables profile discovery. The provisioning workflow begins when a user requests to add a new carrier profile, connecting to the SM-DS+ to discover available profiles, then establishing a secure channel with the SM-DP+ to download and install the selected profile, with mutual authentication and encryption ensuring only authorized profiles can be installed.",
        "expected_sources": ["eSIM_document.txt"],
        "question_type": "technical"
    },
    {
        "question": "What industries benefit from eSIM technology and what are specific use cases?",
        "expected_answer": "While initially popularized in smartphones and consumer electronics, eSIM technology has found applications across various industries. In automotive applications, eSIMs enable connected car services, over-the-air updates, and emergency calling capabilities. The Internet of Things (IoT) sector benefits significantly from eSIMs, particularly in devices that require long-term deployment where physical SIM card replacement would be impractical. Healthcare devices use eSIMs for reliable connectivity in patient monitoring equipment, while wearable technology leverages the space-saving advantages for compact designs.",
        "expected_sources": ["eSIM_document.txt"],
        "question_type": "technical"
    },

    # Roaming Document Questions
    {
        "question": "What is the difference between steered and non-steered roaming?",
        "expected_answer": "In steered roaming, the home network actively guides the subscriber to preferred partner networks through SIM toolkit commands or network signaling. In non-steered roaming, the device selects available networks based on signal strength or other criteria without active guidance from the home network.",
        "expected_sources": ["Roaming_document.txt"],
        "question_type": "comparison"
    },
    {
        "question": "How does SIM-based authentication work in LTE networks for roaming scenarios?",
        "expected_answer": "In LTE networks, SIM-based authentication uses the Authentication and Key Agreement (AKA) protocol. The SIM stores the subscriber key (Ki) and performs the authentication challenge-response with the home network's AuC (Authentication Center), even when roaming, to verify the subscriber's identity and establish a secure connection.",
        "expected_sources": ["Roaming_document.txt"],
        "question_type": "technical"
    },
    {
        "question": "What are the key components of roaming agreements between mobile network operators?",
        "expected_answer": "Roaming agreements between mobile network operators cover pricing, quality of service, technical standards, and settlement procedures for services rendered to roaming subscribers. These agreements define wholesale rates for voice, SMS, and data services, quality of service commitments, technical compatibility requirements, billing and settlement procedures, and fraud management and security provisions.",
        "expected_sources": ["Roaming_document.txt"],
        "question_type": "factual"
    },
    {
        "question": "What security considerations are important in roaming scenarios?",
        "expected_answer": "Security is paramount in roaming scenarios to protect both subscribers and network operators. Authentication security ensures only legitimate subscribers can access services and protects user identity confidentiality. Privacy protection prevents tracking and profiling through temporary identities and encryption of signaling. Fraud prevention systems detect and prevent fraudulent activities through anomaly detection, credit limits, real-time monitoring, and operator collaboration.",
        "expected_sources": ["Roaming_document.txt"],
        "question_type": "technical"
    },
    {
        "question": "How is 5G expected to impact roaming technology?",
        "expected_answer": "5G roaming considerations include network slicing support for differentiated services, enhanced mobile broadband (eMBB) roaming capabilities, ultra-reliable low-latency communication (URLLC) for critical applications, and massive machine-type communications (mMTC) for IoT roaming. The proliferation of eSIM technology affects roaming by enabling easier operator switching for travelers, facilitating multiple profile storage for different regions, supporting remote provisioning of roaming-specific profiles, and reducing physical SIM card logistics for frequent travelers.",
        "expected_sources": ["Roaming_document.txt"],
        "question_type": "technical"
    },

    # LTE/5G Document Questions
    {
        "question": "What are the key differences between LTE and 5G network technologies?",
        "expected_answer": "LTE offers peak data rates of ~300 Mbps with latency of 10-50 ms, while 5G targets peak data rates of up to 20 Gbps with latency of 1 millisecond. 5G provides significantly higher connection density (up to 1,000,000 devices/km² vs ~100,000/km² for LTE), improved spectrum efficiency, and much wider bandwidth capabilities (up to 400 MHz in mmWave vs 20 MHz for LTE). Architecturally, 5G introduces a cloud-native, service-based core network (5GC) replacing LTE's Evolved Packet Core (EPC), and utilizes new technologies like millimeter wave spectrum, massive MIMO, and network slicing.",
        "expected_sources": ["LTE_5G_document.txt"],
        "question_type": "comparison"
    },
    {
        "question": "What are the main architectural components of LTE's Evolved Packet System (EPS)?",
        "expected_answer": "LTE's Evolved Packet System (EPS) consists of User Equipment (UE), Evolved UMTS Terrestrial Radio Access Network (E-UTRAN), and Evolved Packet Core (EPC). The EPC includes the Mobility Management Entity (MME) handling signaling and mobility, Serving Gateway (S-GW) routing user data packets, Packet Data Network Gateway (P-GW) connecting to external networks and handling policy enforcement, and Home Subscriber Server (HSS) storing subscriber profiles and authentication data.",
        "expected_sources": ["LTE_5G_document.txt"],
        "question_type": "factual"
    },
    {
        "question": "What are the key network functions in 5G Core (5GC) and their purposes?",
        "expected_answer": "Key 5G Core network functions include: Access and Mobility Management Function (AMMF) handling access and mobility management, Session Management Function (SMF) managing PDU sessions, User Plane Function (UPF) handling user plane data forwarding, Unified Data Management (UDM) storing subscriber data, Network Repository Function (NRF) enabling service discovery, and Application Function (AF) exposing network capabilities to applications.",
        "expected_sources": ["LTE_5G_document.txt"],
        "question_type": "factual"
    },
    {
        "question": "What deployment strategies do operators use for migrating from LTE to 5G?",
        "expected_answer": "Operators employ various 5G deployment strategies: Greenfield Deployment building new 5G networks in unserved areas, Overlay Approach adding 5G capacity to existing LTE sites, In-Place Upgrade replacing LTE equipment with 5G-capable hardware, Spectrum Refarming converting existing LTE spectrum to 5G use, and Shared Infrastructure utilizing common sites, power, and backhaul for both technologies.",
        "expected_sources": ["LTE_5G_document.txt"],
        "question_type": "technical"
    },
    {
        "question": "What are the three main usage scenarios enabled by 5G technology?",
        "expected_answer": "5G enables three main usage scenarios: Enhanced Mobile Broadband (eMBB) providing higher throughput and capacity, Ultra-Reliable Low-Latency Communication (URLLC) supporting industrial automation, remote surgery, and autonomous vehicles, and Massive Machine-Type Communication (mMTC) enabling smart cities, agricultural sensors, and utility metering applications.",
        "expected_sources": ["LTE_5G_document.txt"],
        "question_type": "technical"
    },

    # SIM Security Document Questions
    {
        "question": "How does the Authentication and Key Agreement (AKA) protocol work in SIM-based authentication?",
        "expected_answer": "The AKA process involves: 1) Challenge Generation where the Authentication Center (AuC) generates a random number (RAND) and computes expected responses, 2) Authentication Tuple creation (RAND, AUTN, XRES, CK, IK), 3) Challenge Delivery sending RAND and AUTN to the visitor network, 4) SIM Computation where the SIM uses the secret key (Ki) to compute RES, CK, IK, and AUTN*, 5) Response Verification where the visitor network forwards RES to home network for comparison with XRES, 6) Mutual Authentication where the SIM verifies AUTN* matches computed AUTN to authenticate the network, and 7) Key Derivation where CK and IK are used to derive ciphering and integrity protection keys.",
        "expected_sources": ["SIM_Security_document.txt"],
        "question_type": "technical"
    },
    {
        "question": "What security mechanisms protect SIM cards against threats?",
        "expected_answer": "SIM security mechanisms include confidentiality protection through ciphering (CK) and integrity (IK) keys derived from AKA, identity protection using temporary identities like TMSI instead of permanent IMSI to prevent tracking, and tamper resistance via secure element hardware with certifications (Common Criteria, FIPS), tamper detection mechanisms, side-channel protection against power/timing attacks, and secure key storage in protected memory areas resistant to extraction.",
        "expected_sources": ["SIM_Security_document.txt"],
        "question_type": "technical"
    },
    {
        "question": "What are the main improvements in 5G AKA (AKA') compared to previous AKA versions?",
        "expected_answer": "5G AKA (AKA') introduces improvements including enhanced protection against IMSI catchers through improved mutual authentication, better privacy protection for the Subscription Concealed Identifier (SUCI), and benefits from the service-based architecture allowing easier security updates and threat response. It maintains backward compatibility while addressing known vulnerabilities in earlier AKA implementations.",
        "expected_sources": ["SIM_Security_document.txt"],
        "question_type": "technical"
    },
    {
        "question": "What future security trends are emerging for SIM technology?",
        "expected_answer": "Emerging SIM security trends include post-quantum cryptography research for quantum resistance, enhanced privacy features like improved SUCI protection and user-controlled privacy controls, and hardware security evolution featuring advanced secure elements, integration with Trusted Execution Environments (TEE), and standardized interfaces for security functionality access. These developments aim to address future threats while maintaining reliability and global compatibility.",
        "expected_sources": ["SIM_Security_document.txt"],
        "question_type": "technical"
    }
]

def get_golden_dataset() -> List[Dict[str, Any]]:
    """Return the golden dataset for evaluation."""
    return GOLDEN_DATASET

def get_questions_only() -> List[str]:
    """Return just the questions from the golden dataset."""
    return [item["question"] for item in GOLDEN_DATASET]

def get_question_by_index(index: int) -> Dict[str, Any]:
    """Get a specific question-answer pair by index."""
    if 0 <= index < len(GOLDEN_DATASET):
        return GOLDEN_DATASET[index]
    raise IndexError(f"Index {index} out of range for dataset of size {len(GOLDEN_DATASET)}")