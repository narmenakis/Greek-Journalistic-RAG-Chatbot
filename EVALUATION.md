# Evaluation

The proposed RAG chatbot was evaluated through a mixed-method approach combining user-centered assessment and system performance measurements. The evaluation aimed to examine both the quality of generated responses and the practical suitability of the application for real newsroom environments.

## Evaluation Setup

- **Participants:** 30 senior Journalism students  
  - 21 female  
  - 9 male  
  - Age: 20–23 years
- **Evaluation Environment:** Controlled computer laboratory sessions
- **Duration:** Two-day study
- **Dataset:** ~73,000 Greek news articles collected from:
  - kathimerini.gr
  - efsyn.gr
  - skai.gr
  - zougla.gr
- **Performance Evaluation Environment:**
  - NVIDIA A100-SXM4-40GB GPU

Participants were already familiar with modern chatbot systems, allowing them to interact naturally with the application and provide realistic evaluations.

---

## Phase 1 – Task-Based Evaluation

Participants used the chatbot to answer predefined journalism-oriented questions.

The evaluation included three thematic topics:

- Tempi railway accident
- 2023 wildfire season
- Daniel storm

Questions were grouped into three levels of cognitive complexity:

1. **Low Complexity – Fact Retrieval**
2. **Medium Complexity – Information Synthesis**
3. **High Complexity – Logical Reasoning & Comparison**

Each response was evaluated using a **0–10 scale** across four dimensions.

### Evaluation Metrics

- **Relevance**  
  Measures how closely the generated answer addresses the user's question.

- **Accuracy**  
  Measures the degree to which the produced information is factually correct.

- **Clarity**  
  Measures the coherence and comprehensibility of the generated answer and retrieved information.

- **Unbiasedness**  
  Measures how neutrally and objectively information is presented without disproportionate emphasis on specific viewpoints.

---

## Phase 1 Results

Overall, the system demonstrated satisfactory performance across all response-quality dimensions.

### Main Findings

- Low- and medium-complexity questions achieved consistently strong scores.
- Performance decreased for reasoning-intensive questions requiring comparison and multi-step synthesis.
- Response quality degradation at higher complexity levels reflects known limitations of current RAG architectures.

### Observations per Metric

#### Relevance and Accuracy
Responses generally remained relevant and factually grounded. Lower-performing cases were primarily associated with retrieval limitations, where semantically similar but less contextually appropriate articles were selected.

#### Clarity
Generated responses maintained strong readability and structure across complexity levels, particularly for retrieval and synthesis tasks.

#### Unbiasedness
Some users observed lower diversity in retrieved sources due to dataset composition, since the majority of articles originated from a single news outlet. This highlights the importance of balanced source coverage in future iterations.

---

## System Performance Evaluation

System execution times were additionally recorded.

### Main Findings

- **Answer generation** was consistently the most computationally expensive stage.
- **Retrieval and summarization** remained highly efficient, typically requiring less than **3 seconds**.
- Execution time increased with:
  - Question complexity
  - Response length
  - Required reasoning depth

### Performance by Complexity

| Complexity Level | Characteristics |
|------------------|----------------|
| Low | Fastest execution times and shortest responses |
| Medium | Moderate execution time with increased synthesis effort |
| High | Highest latency and longest responses |

The evaluation demonstrated a clear relationship between reasoning complexity and computational cost.

---

## Phase 2 – User Experience Evaluation

Participants additionally evaluated the application as an end-user tool.

### Evaluation Metrics

- **Perceived Usefulness**  
  Measures the extent to which users believe the system improves their ability to complete journalism-related tasks more effectively and efficiently.

- **Willingness to Adopt**  
  Measures users’ intention to integrate the application into their regular workflow.

- **User Satisfaction**  
  Measures overall satisfaction with answer quality, usefulness, and system interaction.

- **Search Effectiveness**  
  Measures how effectively the system retrieves relevant supporting information.

- **Ease of Use**  
  Measures how easy and intuitive the application is to operate.

- **Net Promoter Score (NPS)**  
  Measures users’ likelihood of recommending the application to others.

---

## Phase 2 Results

| Metric | Score |
|--------|-------|
| Perceived Usefulness | **8.2 / 10** |
| Adoption Intention | **8.2 / 10** |
| Search Effectiveness | **8.2 / 10** |
| User Satisfaction | **7.9 / 10** |
| Satisfaction with Answers | **8.2 / 10** |
| Satisfaction with Sources | **7.7 / 10** |
| Ease of Use | **8.6 / 10** |
| Tool Attractiveness | **7.5 / 10** |
| No Anxiety / Confusion | **6.5 / 10** |
| Net Promoter Score | **50 (Excellent)** |

### Key Conclusions

- Participants perceived the system as useful, easy to use, and effective for journalistic information retrieval.
- Source-grounded answer generation increased confidence in the produced responses.
- The strongest performance was observed in factual retrieval and synthesis tasks.
- Users expressed a strong willingness to recommend and adopt the application in newsroom workflows.

---

## Limitations and Future Improvements

Several areas remain open for improvement:

- Increasing source diversity across news outlets
- Improving reasoning capabilities for high-complexity questions
- Optimizing generation latency
- Expanding evaluation to broader user groups
- Supporting larger-scale multi-user deployments

Overall, the results indicate that the proposed RAG-based system has strong potential as an assistive tool for archive exploration, source-grounded information retrieval, and evidence-based journalism.
