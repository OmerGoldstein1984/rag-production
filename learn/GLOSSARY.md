# RAG Academy Glossary

## Key Terms and Concepts

## A
- **Accuracy**: The degree to which the system's responses match factual information from source documents
- **Augmentation**: The process of combining retrieved context with the user's query to create an input for the LLM
- **Average Precision**: Measures the quality of ranked retrieval results by averaging precision at each relevant document position
- **Access Control List (ACL)**: A list of permissions attached to an object that specifies which users or system processes are granted access to objects, as well as what operations are allowed on given objects
- **Anonymization**: The process of removing or modifying personal information in data so that the individuals whom the data describe cannot be identified

## B
- **Batch Processing**: Processing multiple items (texts, queries) together for efficiency
- **Bi-encoder**: A model that encodes queries and documents separately for retrieval
- **BM25**: A ranking function used in information retrieval that measures term frequency-inverse document frequency
- **Bias (in LLMs)**: Systematic preference for certain outputs over others, such as position bias, self-preference bias, or verbosity bias

## C
- **Chunking**: The process of splitting documents into smaller pieces for embedding and retrieval
- **Context Window**: The maximum amount of text (measured in tokens) that an LLM can process at one time
- **Cosine Similarity**: A measure of similarity between two vectors based on the cosine of the angle between them
- **Cross-encoder**: A model that encodes query-document pairs together for more accurate relevance scoring
- **Citations**: Explicit references to source documents that support parts of the generated answer
- **Conversational Memory**: System that maintains context across multiple turns enabling follow-up questions to be understood in context
- **HyDE (Hypothetical Document Embedding)**: Technique that generates a hypothetical document answering the query, then uses its embedding for retrieval

## D
- **Dense Retrieval**: Retrieval method using neural embeddings to find semantically similar vectors
- **Dimension**: The number of values in an embedding vector
- **Dot Product**: A mathematical operation that returns a single number from two equal-length sequences of numbers
- **Data Freshness**: How recently the information in a system was updated or validated
- **Data Retention Policy**: A protocol for how long information is kept and how it's disposed of

## E
- **Embedding**: A numerical representation of text that captures semantic meaning
- **Evaluation**: The process of assessing the quality and performance of a RAG system
- **Faithfulness**: The degree to which a generated answer is supported by the retrieved context
- **FDE**: Field Development Engineer - the target audience for this RAG system as interview preparation

## F
- **Fine-tuning**: Continuing to train a pre-trained LLM on a specific dataset to adapt it to a domain
- **Fallback Mechanism**: A backup system or procedure that takes over when the primary system fails
- **Feedback Loop**: A system where outputs are routed back as inputs to improve future performance

## G
- **Generation**: The stage where the LLM produces the final answer based on the augmented prompt
- **Grounding**: Ensuring the LLM's response is based solely on the provided context
- **Graceful Degradation**: The ability of a system to maintain limited functionality even when part of the system fails

## H
- **Hallucination**: When an LLM generates information that sounds plausible but is factually incorrect or not present in the input context
- **Hybrid Retrieval**: Combining dense and sparse retrieval methods to get benefits of both
- **Hallucination Rate**: The percentage of generated answers that contain information not supported by the retrieved context

## I
- **Ingestion**: The process of collecting and preparing source documents for processing
- **Inference**: The process of running a model to predict outputs or generate responses
- **Idempotent**: A property of operations where applying them multiple times has the same effect as applying them once
- **Observability**: The ability to measure the internal states of a system by examining its outputs
- **On-Premises**: Software that is installed and runs on computers on the premises of the person or organization using the software

## J
- **JWT (JSON Web Token)**: An open standard for securely transmitting information between parties as a JSON object

## K
- **Knowledge Cutoff**: The point in time after which an LLM lacks knowledge about events or information
- **Key-value Store**: A data storage paradigm designed for storing, retrieving, and managing associative arrays
- **Knowledge Base**: A centralized repository for information: a public library, a database of related information about a particular subject

## L
- **LLM (Large Language Model)**: A deep learning model trained on vast amounts of text data to understand and generate human-like language
- **Latency**: The delay between a query being made and a response being received
- **Language Model**: A probabilistic model of a natural language that can generate probabilities of a series of words
- **Latent Semantic Analysis (LSA)**: Technique that identifies relationships between terms and concepts in text using singular value decomposition
- **Load Balancing**: Distributing workloads across multiple computing resources to optimize resource use

## M
- **MRR (Mean Reciprocal Rank)**: A statistic measure for evaluating any process that produces a list of possible responses to a sample of queries
- **Metadata**: Data that provides information about other data
- **Mock Provider**: A simulated implementation of a provider interface for testing and development
- **MRR**: Mean Reciprocal Rank - measures how highly ranked the first relevant document is
- **Multi-query Expansion**: Technique that generates multiple query variations to improve recall and capture different aspects of information need
- **Microservices**: An architectural style that structures an application as a collection of loosely coupled services
- **Mutation Testing**: A form of software testing where certain statements of the source code are changed/mutated to check if the test cases are able to find the errors

## N
- **Nearest Neighbor Search**: Finding the closest points to a given point in a metric space
- **Normalization**: The process of adjusting values measured on different scales to a notionally common scale
- **NLU (Natural Language Understanding)**: A subtopic of natural language processing in artificial intelligence that deals with machine reading comprehension

## O
- **Overlap**: Amount of text shared between adjacent chunks to prevent information loss at boundaries
- **Orchestration**: The coordination and management of multiple components in a system
- **Open Source**: Software with source code that anyone can inspect, modify, and enhance
- **Operational Excellence**: A philosophy of leadership, teamwork, and problem-solving resulting in continuous improvement throughout the organization

## P
- **Precision**: The fraction of retrieved instances that are relevant
- **Prompt**: The input text given to a language model to generate a response
- **Prompt Engineering**: The process of structuring text that can be interpreted and understood by a generative AI model
- **Prompt Stuffing**: Putting as much relevant information as possible directly into the LLM's prompt
- **Prompt Injection**: A security concern where malicious actors attempt to manipulate the LLM's behavior through carefully crafted inputs
- **Provider Interface**: An abstract base class that defines the contract for LLM or embedding providers
- **RAG (Retrieval-Augmented Generation)**: Combining a retrieval system with a generation system to enhance LLM responses with external knowledge
- **Recall**: The fraction of relevant instances that were retrieved
- **Reciprocal Rank Fusion (RRF)**: Technique for combining multiple ranked lists using the formula: score = Σ 1/(k + rank_i)
- **Retrieval**: The process of finding relevant information from a knowledge base
- **Reranking**: Re-scoring retrieved results with a more accurate (but slower) model
- **Reranker**: A model used to re-score retrieved results for better relevance
- **Rate Limiting**: A technique for limiting network traffic to prevent overload
- **Root Cause Analysis**: A method of problem solving used for identifying the root causes of faults or problems

## Q
- **Query Rewriting**: Process of transforming user queries into forms that better match document content
- **Query Expansion**: Technique that expands the original query with additional terms to improve retrieval
- **Quality Assurance (QA)**: A way of preventing mistakes and defects in manufactured products and avoiding problems when delivering solutions or services to customers

## R
- **Relevance Feedback**: Process where users indicate which results are relevant, and the system uses this to improve future queries
- **Retrieval Augmented Generation (RAG)**: Combining a retrieval system with a generation system to enhance LLM responses with external knowledge

## S
- **Sparse Retrieval**: Traditional information retrieval using term frequency-inverse document frequency (e.g., BM25)
- **Similarity Metrics**: Methods for measuring how alike two vectors are (cosine similarity, dot product, euclidean distance)
- **Student**: In the context of LLM analogies, refers to a model that has learned from training data
- **Temperature**: A parameter that controls the randomness of LLM output
- **Token**: The basic units of text that LLMs process (can be words, parts of words, or characters)
- **Top-k**: The number of most similar items to return from a retrieval operation
- **Throughput**: The amount of material or items passing through a system or process
- ** Table of Contents (TOC)**: A list, usually found on a page before the start of a writing, that shows what the document contains

## T
- **Token Bucket Algorithm**: A rate limiter that allows a burst of traffic for a short time while limiting the long-term rate
- **Tracing**: Tracking the execution path of a request through a system
- **Turnaround Time (TAT)**: The amount of time taken to complete a process or fulfill a request

## U
- **Uptime**: The amount of time that a service is available and operational
- **User Acceptance Testing (UAT)**: A phase of software development where the software is tested by the intended audience in real world conditions

## V
- **Vector**: A quantity having direction as well as magnitude, especially as determining the position of one point in space relative to another
- **Vector Space**: A mathematical structure formed by a collection of vectors
- **Vector Store**: A specialized database for storing and querying embedding vectors
- **Version Control**: The management of changes to documents, computer programs, large web sites, and other collections of information

## W
- **Weighted Combination**: Combining multiple scores or values using weights to reflect their relative importance
- **Whiteboard Design**: A collaborative planning session where participants draw diagrams and write concepts on a whiteboard to work through system design
- **Workflow**: An orchestrated and repeatable pattern of business activity enabled by the systematic organization of resources into processes that transform materials, provide services, or process information

## EVALUATION TERMS
- **Hit Rate@K**: The proportion of queries for which at least one relevant document appears in the top K retrieved results
- **Recall@K**: The proportion of relevant documents that are retrieved in the top K results
- **MRR (Mean Reciprocal Rank)**: The average of the reciprocal ranks of the first relevant document for each query
- **nDCG@K (Normalized Discounted Cumulative Gain at K)**: Measures the usefulness of a document based on its position in the result list
- **LLM-as-Judge**: Using a language model to evaluate generation quality with predefined rubrics
- **Golden Set**: A benchmark dataset of questions with expected answers and source documents for evaluation
- **Ablation Study**: Systematic removal or modification of components to understand their individual contributions
- **Faithfulness**: The degree to which a generated answer is supported by the retrieved context
- **Answer Relevance**: Measures how well the generated answer addresses the original question
- **Answer Completeness**: Measures whether the answer contains all expected information elements

## PRODUCTION TERMS
- **Canary Deployment**: A technique to reduce the risk of introducing a new software version in production by slowly rolling out the change to a small subset of users before rolling it out to the entire population
- **Circuit Breaker**: A design pattern used to detect failures and encapsulate the logic of preventing a failure from constantly recurring during maintenance, temporary external system failure or unexpected system difficulties
- **Content Delivery Network (CDN)**: A geographically distributed network of proxy servers and their data centers
- **Data Minimization**: The practice of limiting the collection of personal information to that which is directly relevant and necessary to accomplish a specified purpose
- **Disaster Recovery (DR)**: An organization's method of regaining access and functionality to its IT infrastructure after events like a natural disaster, cyber attack, or even business disruptions
- **Environment Variables**: Dynamic-named values that can affect the way running processes will behave on an operating system
- **Extract, Transform, Load (ETL)**: The general procedure of copying data from one or more sources into a destination system which represents the data differently from the source(s)
- **Fault Tolerance**: The property that enables a system to continue operating properly in the event of the failure of some (or one or more) of its components
- **Golden Signal**: Key metrics that indicate the health of a system (latency, traffic, errors, saturation)
- **Health Check**: A test performed to determine if a system or component is operating correctly
- **Idempotency Key**: A unique identifier used to ensure that a request is processed only once
- **Infrastructure as Code (IaC)**: The process of managing and provisioning computer data centers through machine-readable definition files
- **Latency Budget**: The maximum amount of time allowed for a particular operation or process
- **Load Shedding**: The deliberate shutdown of electric power in a part or parts of a power-distribution system, generally to prevent the failure of the entire system when the demand strains the capacity of the system
- **Logging**: The process of recording events that occur in a system or organization
- **Log Aggregation**: The process of collecting log data from multiple sources and storing it in a centralized location
- **Message Queuing**: A form of asynchronous service-to-service communication used in serverless and microservices architectures
- **Monitoring**: The process of observing and checking the behavior and outputs of a system and its components over a period of time
- **Mutual TLS (mTLS)**: A method for mutual authentication to ensure that both parties in a communication are who they claim to be
- **Observability**: The ability to measure the internal states of a system by examining its outputs
- **Orphaned Process**: A computer process that has been initiated but whose parent process has terminated or ended
- **Point-in-Time Recovery (PITR)**: The ability to restore a database to a specific point in time
- **Policy as Code**: The practice of managing policies through code and software development techniques
- **Quality of Service (QoS)**: The description or measurement of the overall performance of a service, such as a telephony or computer network or cloud computing service
- **Rate Limiting**: A technique for controlling the rate of requests sent or received by a network interface controller
- **Replication**: The process of sharing information so as to ensure consistency between redundant resources
- **Resource Utilization**: The amount of resources used by a system or process
- **Rollback**: The process of returning a system to a previous state by removing the last set of changes
- **Service Level Agreement (SLA)**: A commitment between a service provider and a client
- **Service Level Indicator (SLI)**: A carefully defined quantitative measure of some aspect of the service
- **Service Level Objective (SLO)**: A target value or range of values for a service level indicator that is agreed upon in an SLA
- **Sharding**: A database partitioning technique that separates very large databases into smaller, faster, more easily managed parts called data shards
- **Single Point of Failure (SPOF)**: A part of a system that, if it fails, will stop the entire system from working
- **Snapshot**: A copy of a set of data at a specific point in time
- **Software Bill of Materials (SBOM)**: A formal, publicly available, structured list of components, libraries, and modules that are required to build a piece of software
- **Stateful vs Stateless**: Whether a system remembers previous interactions or treats each request as independent
- **Timeout**: A parameter that specifies an amount of time that should be allowed for an event to occur
- **Tokenization**: The process of breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols, and other elements called tokens
- **Tracing**: Tracking the execution path of a request through a system
- **Versioning**: The assignment of either unique version names or unique version numbers to unique states of computer software
- **Warm Standby**: A method of redundancy in which the secondary system runs in the background, mirroring the primary system
- **Zero Trust**: A security concept centered on the belief that organizations should not automatically trust anything inside or outside its perimeters and instead must verify anything and everything trying to connect to its systems

## INTERVIEW PREPARATION TERMS
- **STAR Method**: A technique used to answer behavioral interview questions by describing the Situation, Task, Action, and Result
- **Whiteboard Interview**: A type of technical interview where the candidate is asked to solve problems or design systems on a whiteboard
- **System Design Interview**: An interview focused on assessing a candidate's ability to design scalable and robust systems
- **Behavioral Interview**: An interview focused on assessing a candidate's past behavior in various situations to predict future performance
- **Technical Screening**: An initial interview to assess a candidate's basic technical qualifications
- **Culture Fit**: The likelihood that a candidate will be able to conform and adapt to the core values and collective behaviors that make up an organization
- **Elevator Pitch**: A brief, persuasive speech that you use to spark interest in what your organization does
- **Use Case**: A description of how a user will interact with a system to achieve a particular goal
- **Edge Case**: A problem or situation that occurs only at an extreme (maximum or minimum) operating parameter
- **Regression Testing**: A type of software testing that ensures that recently developed and tested code still performs correctly after it was interfaced with existing code
- **Smoke Test**: A preliminary test to reveal simple failures severe enough to cause a rejection
- **A/B Testing**: A way to compare two versions of something to determine which one performs better
- **Mock Interview**: A practice interview that simulates the real interview experience
- **Imposter Syndrome**: The persistent inability to believe that one's success is deserved or has been legitimately achieved as a result of one's own efforts or skills