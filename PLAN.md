# Discover Agent Plan

This project is for learning AI engineering end to end, as well as building a
working system. Work through the tasks one at a time. Unchecked tasks are pending;
mark them done as they finish and record relevant decisions or results here.

## Done

* Ingestion from BBC Business RSS with deduplication (ingest.py)
* Body cleaning for footers and reader callouts (clean_body)
* Article embeddings: title plus first 1500 characters, 768 dims, normalized (embed.py)
* Raw vector search (similar.py)
* Two stage link reranker with v3 prompt, usefulness scores, deterministic selection, retries and fallback model (rerank.py)
* Unit tests (tests/) and a multi run link eval (eval_links.py, eval/link_cases.json)

## Finish the Link Researcher

- [ ] **1. Settle the Ed Davey question.** Compare the usefulness scores for
  articles 9 and 15, then either adjust the prompt or move article 15 in the golden
  set. Record the reasoning behind the decision.
- [ ] **2. Grow the golden set.** Expand it from 3 cases to about 10, using the
  newer articles in the corpus.
- [x] **3. Define when to return no links.** Decide a clear rule for when the
  Link Researcher should return no links at all.
  Result: implemented floors of same_story usefulness >= 3 and related/explainer
  usefulness >= 4; return no links instead of padding below those floors. Eval
  note: case 22 should-link article 21 dropped from 3/3 before to 1/3 after.

## Build the other agents

- [ ] **4. Build the Headline Writer.** Add a versioned prompt, a Pydantic schema,
  and plain code checks for length and style rules before anything reaches a
  model reviewer.
- [ ] **5. Build the Tagger.** Define a controlled tag vocabulary and use it for
  tagging. It may be folded into another agent if it does not earn its place.
- [ ] **6. Build the Reviewer.** Check every suggestion against the article and
  the rules, and reject suggestions with specific reasons.

## Connect them

- [ ] **7. Build the orchestrator and shared state.** Add a classifier that
  decides the article type: breaking, explainer, long form, or standard.
- [ ] **8. Write the graph executor by hand.** Support conditional edges,
  parallel branches, and a fan in step before review.
- [ ] **9. Add the review loop.** Send each rejected item back to its agent with
  the rejection reasons, capped at two retries, with a cost and iteration budget
  per run.

## Make it usable

- [ ] **10. Persist runs and decisions.** Add tables for runs, suggestions, and
  editor decisions in numbered migrations starting at `sql/003_...`, so every
  accept and reject is logged.
- [ ] **11. Build a small React page.** Show the article on one side and
  suggestions on the other, with accept and reject buttons and a trace panel
  showing which path the article took.
- [ ] **12. Schedule ingestion and embedding.** Add a trigger so ingest and
  embed run on a schedule instead of by hand.

## Measure and finish

- [ ] **13. Evaluate headlines and the Reviewer.** I will rate
  generated and original headlines blind. Track how often the Reviewer catches
  a real problem versus rejecting something fine.
- [ ] **14. Track cost and latency.** Record cost and latency per run.
- [ ] **15. Deploy and document the project.** Complete deployment, a README,
  and a demo that I can walk someone through.

## Later, once decision data exists

- [ ] **16. Explore classical ML and fine tuning.** Build the classical ML
  predictor and the fine tuning experiment once enough accept and reject data
  has been collected. Both depend on the decision log from task 10.

## Learning responsibilities

- **Codex can do most of the implementation:** code and tests for the no-links
  rule (3), once I have decided the rule; database tables (10), the React
  page (11), scheduled ingest and embed (12), cost and latency tracking (14), and
  deployment and the README (15).
- **I should lead or rewrite:** the golden set and link quality
  decisions (1–2), and the prompts for the Headline Writer, Tagger, and Reviewer
  (4–6). I decide what a good link is; prompts are central to
  each agent's behavior.
- **Sketch the graph before implementation:** I should design
  the state object, nodes, and edges (7–8), then read every line of the code Codex
  writes.
- **Keep headline ratings human:** my blind judgment supplies
  the evidence for task 13.

For every task, I should be able to explain how the result works
in an interview. If that is not possible, I should slow down and read the code before
moving on.

## Working rules for Codex

* Do only the task I name. Do not start the next one.
* Never change prompt files for tasks marked as mine unless I ask.
* After finishing, update the checkbox here and note any decision or result in one line.
* If a task needs a design choice that is mine to make, ask me before coding.
