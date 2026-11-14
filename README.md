# CarePath AI — Healthcare Conversion Intelligence

**Understand patient journeys. Discover growth opportunities. Take smarter action.**

> **Disclaimer**
> Independent personal portfolio project. Uses synthetic demonstration data only.
> Not intended for clinical decisions or processing real patient information.
>
> This prototype is not HIPAA compliant, has not been security reviewed, and is
> not production ready. It processes no protected health information because it
> processes no real data at all.

CarePath AI is an AI-powered healthcare marketing intelligence copilot. It helps
a marketing team see where visitors abandon the appointment-booking journey,
form hypotheses about why, and decide which growth experiment is worth running.

It answers three questions:

1. **Where are we losing potential appointments?**
2. **What might explain the conversion problem?**
3. **What should we test, and what is its potential financial impact?**

---

## Quick start

```bash
git clone <your-fork-url> && cd PatientCare
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501` and works immediately — **no API key is
required**. With no key configured, the AI Conversion Copilot answers from a
deterministic rule-based analyst using the same evidence, and makes no network
calls at all.

To enable the live model:

```bash
export OPENAI_API_KEY="sk-..."      # or copy .streamlit/secrets.toml.example
streamlit run app.py
```

Run the tests:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## 1. Why I chose to build this

I wanted a project that forced me across the whole surface of an AI product
rather than just the model call: generating a believable dataset, computing
analysis that holds up to scrutiny, deciding what an AI layer should and should
not be trusted with, designing an interface an executive could actually read,
and building the guardrails that make an AI recommendation safe to act on.

Healthcare marketing turned out to be a good setting for that, because the
stakes make the guardrails obvious. A marketing funnel is commercially
meaningful, the data is genuinely messy, and the moment you touch anything
patient-adjacent the privacy question stops being theoretical. That let me
design *for* constraints instead of bolting them on at the end.

This is my own project. It is not an employer project, a client project, or a
commercial healthcare platform, and it is not modelled on any specific company's
product.

## 2. The healthcare marketing problem I identified

A hospital's website attracts a large volume of visitors who show real intent —
they search for a provider, they open the booking flow — and then disappear.
Marketing teams usually have clear visibility of the two ends of this: traffic
at the top, booked appointments at the bottom. What they often cannot see is
*where* intent is lost, *which segment* is losing it, and *what a fix would be
worth*.

Analytics tools are good at "what happened". Someone still has to turn that into
"what should we test, and is it worth the engineering time". That gap — between
a dashboard and a decision — is where I aimed this product.

## 3. My product hypothesis

**A marketing team will act on an AI recommendation only if it can see the
arithmetic behind it.**

So the central design bet is a strict separation of responsibilities:
**deterministic code computes every number, and the model is only allowed to
interpret them.** The model receives an aggregated evidence pack and a system
prompt that forbids inventing figures. Afterwards, every figure in its answer is
checked against that pack, and anything that cannot be traced back is flagged in
the interface.

That constraint costs some conversational fluency. In exchange:

- every claim is auditable,
- the product works with no API key at all,
- and a user who distrusts the AI can re-derive its numbers from the dashboard.

## 4. The users I designed it for

- Hospital marketing managers
- Digital health growth teams
- Healthcare marketing analysts
- Patient acquisition teams

Concretely: someone who owns a conversion number, has to defend a budget, and
has very limited engineering time to spend. The product's real job is helping
them decide *what not to build*.

## 5. MVP scope and product decisions

The MVP is four modules:

| Module | What it does |
| --- | --- |
| **Growth Dashboard** | KPI cards, booking funnel, drop-off waterfall, segment comparison, channel economics, daily trends, and ranked opportunity sizing, all filterable by date, channel, device and service line. |
| **AI Conversion Copilot** | Natural-language questions answered from an aggregated evidence pack, with mode badges, evidence validation, a model input preview and the full prompt on display. |
| **AI Experiment Studio** | Turns a measured gap into a hypothesis, a powered A/B design, guardrail metrics, an explicit financial projection, a reproducible simulator, and a human approval workflow with an audit trail. |
| **AI Trust & Privacy** | Privacy principles, the outbound allowlist, the exact payload, data-quality and missing-data reporting, a live input screener, and a runnable evaluation suite. |

I deliberately kept four things **out** of the MVP, and each exclusion was a
product decision rather than a shortcut:

- **No real data connectors.** A connector would have made the demo more
  impressive and the privacy story much worse. Synthetic data let me design the
  egress controls honestly.
- **No agentic actions.** The copilot proposes; a named human approves. Nothing
  launches itself. For anything patient-adjacent I think that is the correct
  default, not a limitation.
- **No causal claims.** The product measures gaps against comparable peer
  segments. It never asserts that a cause has been established, because this
  data cannot establish one.
- **No invented numbers.** This single rule drove the whole architecture,
  including a fallback analyst good enough to be the default experience.

## 6. The technical architecture

```
app.py              Streamlit presentation layer — no business logic
├── data_generator.py   seeded synthetic dataset with deliberately planted signals
├── analytics.py        every metric, rate, significance test and projection
├── privacy.py          outbound allowlist, PII screening, data-quality reporting
├── ai_analyst.py       prompt construction, model call, fallback, verification
├── experiment.py       design, power analysis, projection, simulation, approvals
└── evaluation.py       analytics invariants and AI behaviour checks
tests/                  168 automated tests across all six modules
```

The dependency direction is one-way: `app.py` imports everything, and nothing
imports `app.py`. The analytics layer has no knowledge of the AI layer, which is
exactly what makes the AI layer replaceable.

**The dataset.** 90 days × 6 channels × 2 devices × 5 service lines = 5,400 rows,
generated from a fixed seed so every reviewer sees identical numbers. Each funnel
stage is drawn with a binomial from the stage above it, so the funnel is
internally consistent by construction. The world contains a few deliberately
planted signals for the analytics and AI layers to rediscover: a large mobile
booking-completion gap, a weak paid-social channel, a mid-period campaign ramp,
weekday seasonality, and about 1% missing spend values.

**Opportunity sizing.** For a given funnel step, the highest-converting segment
with enough volume becomes the benchmark. For every other segment the app
computes the conversions that would occur at parity, flows them through that
segment's *observed* downstream rates to attended appointments, and values them
at its observed contribution margin. A capture-rate control scales the gap, so a
user can plan against a realistic share rather than full parity.

**Determinism.** The dataset carries a signature — a hash of its funnel counts —
shown in the sidebar. The analytics layer has no randomness, and the only
stochastic component in the product, the experiment simulator, is seeded and
exposes its seed in the UI.

## 7. The AI integration and its limitations

The model receives roughly ten kilobytes of aggregated metrics plus a system
prompt that requires it to separate observation from hypothesis, state
uncertainty, and recommend one specific action with a guardrail metric.
Temperature defaults to 0.2.

The copilot runs in one of four modes, and the UI always says which:

| Mode | When | Behaviour |
| --- | --- | --- |
| `llm` | A key is configured and the call succeeded | The model's answer, grounded in the evidence pack |
| `fallback` | No key configured, or offline mode forced | Deterministic rule-based answer from the same pack; nothing transmitted |
| `error-fallback` | A key is configured but the call failed | Deterministic answer, with the error surfaced rather than swallowed |
| `blocked` | The question failed privacy screening | Nothing is sent; the user is told why |

**Known limitations, stated plainly:**

- The verifier matches figures numerically. It cannot catch a *correctly quoted
  number used to support a wrong conclusion*.
- The evidence pack is a fixed schema. The model cannot ask a follow-up question
  of the data, so a question outside the pack's coverage gets a partial answer.
- The fallback analyst is rule-based. It is reliable and reproducible, but it
  only recognises the intents I wrote handlers for; anything else gets a general
  overview.
- The dataset itself has no page-level events, no form-field analytics, no
  session recordings and no patient outcomes. Several hypotheses the copilot
  raises are, by design, untestable with the data in scope — and it says so.
- I have evaluated the copilot's *behaviour*, not its *judgement*. Whether its
  recommendations are good is not something this harness can tell you.

## 8. The data privacy approach

- **Synthetic data only.** No database, no upload path, no connection to any real
  analytics or clinical system.
- **Aggregates leave, records never do.** The copilot receives funnel rates,
  segment totals and projected values. Row-level data is never serialised into a
  prompt.
- **Allowlist by construction.** The outbound payload is *rebuilt* from an
  explicit field allowlist rather than filtered, so a new upstream field is
  excluded by default until someone deliberately adds it. Anything dropped is
  reported in the UI.
- **Input screening.** Questions are scanned for patterns resembling personal or
  health identifiers, and for phrases requesting individual-level information. A
  match blocks the request outright rather than silently redacting it.
- **Inspect before you send.** The exact JSON payload is shown before any request
  is made and after every answer, and can be downloaded.
- **Final backstop.** The serialised payload is re-scanned for PII patterns
  immediately before transmission.
- **Human approval.** Experiments stay in Draft until a named reviewer approves
  them, and every transition is recorded.
- **No secrets in source.** Keys are read from `st.secrets` or `OPENAI_API_KEY`,
  masked wherever they are displayed, and a test asserts that no key literal
  appears in the source.

**This is privacy-aware design, which is not the same as compliance.** No
compliance claim is made anywhere in this project.

## 9. How the product's value could be measured

I have not run this with users, so these are the measures I would set up — not
results I have:

- **Decision quality** — what share of approved experiments actually ship, and
  what share of shipped ones beat their guardrails.
- **Projection calibration** — projected lift against measured lift. This is the
  number that would tell me whether the sizing model is honest, and the one I
  would most want to track.
- **Trust behaviour** — how often users open the evidence panel, and whether
  their reliance on the copilot changes once they have.
- **Time to a decision** — from question asked to experiment approved.
- **Grounding rate in production** — the share of answers passing evidence
  validation with no flagged figures.

## 10. Future development roadmap

| Theme | Next step |
| --- | --- |
| Data | Connectors to real analytics sources, with a privacy review *before* any is built |
| Data | Page-level and form-field event data — the single biggest gap in the current funnel |
| Analysis | Causal methods (difference-in-differences, matched segments) instead of parity gaps |
| Analysis | Multi-touch attribution rather than the last-touch model baked into this dataset |
| Experimentation | Result ingestion, so projections can be scored against outcomes |
| Experimentation | Sequential testing with proper alpha spending, to make early stopping safe |
| Platform | Persistent storage and multi-user roles for the approval workflow |
| AI | A model-evaluation harness comparing providers on the same evidence pack |

---

## Implemented vs planned

**Implemented and running in this application today:**

- Deterministic 90-day synthetic marketing dataset (5,400 rows, fixed seed)
- Funnel, segment, channel-economics, trend and period-comparison analytics
- Parity-gap opportunity sizing with an adjustable capture-rate assumption
- Two-proportion significance testing on segment differences
- AI copilot grounded in an aggregated evidence pack, with a full offline fallback
- Post-hoc evidence validation that flags figures the evidence cannot support
- Allowlist-based egress control, PII screening and a model input preview
- Experiment designer with powered sample size, runtime estimate and guardrails
- Financial projection engine with explicit, user-adjustable assumptions
- Synthetic A/B simulator with reproducible seeds
- Human approval workflow with an audit trail
- Automated evaluation suite runnable from inside the product

**Planned, not built** — everything in the roadmap table above. Nothing in that
column exists yet, and this project does not claim otherwise anywhere.

---

## Testing

```bash
pytest                      # full suite
pytest tests/test_privacy.py -v
python evaluation.py        # the AI evaluation suite from the command line
```

168 tests cover dataset invariants and determinism, analytics correctness and
edge cases, the privacy allowlist and screener, AI grounding and fallback
behaviour (including a mocked model, a hallucinated figure and a failed call),
experiment power/projection/simulation/approval logic, and the evaluation
harness itself. The evaluation suite is also runnable from the AI Trust &
Privacy module inside the app.

## Deployment to Streamlit Community Cloud

1. Push this repository to GitHub. `.streamlit/secrets.toml` is git-ignored, so
   no key can be committed.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing
   at `app.py` on your branch.
3. **Optional** — add secrets in the app's settings:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   ```
4. Deploy. Without secrets the app still runs end to end in deterministic
   fallback mode, which is the intended default for a reviewer.

The dataset is generated at runtime from a seed, so there is no data file to
upload and no storage to provision.

## Project structure

```
PatientCare/
├── app.py                      Streamlit UI (5 pages)
├── data_generator.py           synthetic dataset
├── analytics.py                deterministic metrics and projections
├── ai_analyst.py               AI copilot, fallback and verification
├── experiment.py               experiment design, power, projection, approvals
├── privacy.py                  allowlist, screening, data quality
├── evaluation.py               evaluation harness
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .streamlit/
│   ├── config.toml             theme
│   └── secrets.toml.example    template — the real file is git-ignored
└── tests/                      168 tests
```

## Licence and attribution

Independent personal portfolio project by **<add your name here>**. All data is
fictional and generated by `data_generator.py`. No real organisation, patient or
visitor is represented.
