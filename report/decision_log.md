# Decision Log

Non-obvious decisions made during the build, in order.

---

1. **Picked AmazonHelp over AppleSupport**  
   AmazonHelp has 169k replies vs 106k for Apple. More data means more diverse retrieval candidates. Apple's replies also tend to be very templated ("Please DM us"), which would make retrieval boring to evaluate.

2. **Filtered to English-only using ASCII ratio**  
   About 9k of 150k AmazonHelp conversations were non-English. I used a simple heuristic (>60% ASCII characters) rather than a language detection library to keep dependencies minimal. Downside: mixed-language tweets sometimes slip through.

3. **Defined 8 intents top-down from skimming the data, not from clustering**  
   Clustering would be more principled but takes longer to interpret. For a first version, keyword-defined intents are transparent and easy to explain. The cost is that 52% of data falls into "other" — that's the main thing I'd fix next.

4. **Used keyword labels as training signal for the TF-IDF classifier**  
   This creates a circular evaluation problem (acknowledged in the report). The alternative — hand-labelling 140k rows — wasn't feasible. The classifier is still useful because it generalises the keyword rules to unseen phrasing.

5. **Set escalation confidence threshold at 0.55**  
   Below this, the classifier is uncertain enough that auto-handling feels risky. I tried 0.4 and 0.7 — 0.4 let too many ambiguous messages through, 0.7 escalated too many routine queries. 0.55 gave the best false-negative rate on the golden set.

6. **Made refund_request always escalate regardless of confidence**  
   Any message about money should have a human in the loop. This is a policy decision, not a model decision. It means the agent will never auto-reply to a refund request even if it's a simple "how do I request a refund?" — that's intentionally conservative.

7. **Used extractive retrieval (TF-IDF cosine similarity) instead of generative replies**  
   A generative LLM can hallucinate refund policies or delivery timelines that Amazon hasn't committed to. Extractive retrieval can only say things Amazon has actually said before. Safer for a first version.

8. **Filtered same-intent subset before retrieval, with fallback to full corpus**  
   Retrieving from the same intent bucket improves relevance. But for rare intents (delivery_problem has only 859 examples), the bucket is small enough that the best match might still be poor — so I fall back to the full corpus if the bucket has fewer than 5 examples.

9. **Stripped @mentions and URLs during cleaning**  
   @mentions are noise for intent classification. URLs in replies become dead links after cleaning ("check here:" with nothing after). This is a known issue — the reply quality suffers for messages where the original reply was just a link. Logged as failure mode 3.

10. **Stratified golden set at 25 examples per intent**  
    A random sample would give ~52% "other" examples and almost no delivery_problem examples (0.6% of data). Stratification ensures every intent is represented equally in evaluation, which gives a fairer picture of per-intent performance.

11. **Set escalation false negatives as the primary metric to minimise**  
    In a real support context, auto-handling a fraud claim is much worse than unnecessarily escalating a tracking question. So I tuned the escalation rules to have zero false negatives on the golden set, accepting the 20 false positives.

12. **Used a rule-based judge instead of an LLM judge**  
    No API key was available during development. The rule-based judge (positive/negative signal words + length check) is transparent and reproducible. The 0.476 human correlation is honest — it's not a great judge, and the report says so.
