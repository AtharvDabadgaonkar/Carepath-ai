# CarePath AI

**Healthcare Conversion Intelligence.** Understand patient journeys, find growth opportunities, take smarter action.

> **Disclaimer**
> Independent personal portfolio project. Uses synthetic demonstration data only.
> Not intended for clinical decisions or processing real patient information.
>
> It isn't HIPAA compliant, hasn't been security reviewed, and isn't production
> ready. It handles no protected health information, for the simple reason that
> it handles no real data at all.

A hospital's website pulls in people who clearly want an appointment. They search
for a provider, they open the booking form, and then a lot of them vanish.
Marketing teams can usually see the two ends of that story (traffic at the top,
bookings at the bottom) but not the middle, where the actual problem lives.

CarePath AI is my attempt at the middle. It's an AI copilot that answers three
questions for a marketing team:

1. Where are we losing appointments?
2. What might explain it?
3. What should we test, and what's it worth?

---

## Try it

```bash
git clone https://github.com/AtharvDabadgaonkar/Carepath-ai.git
cd Carepath-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501` and **you don't need an API key**. Without
one, the copilot falls back to a rule-based analyst that reads the same evidence
and makes no network calls whatsoever. That's the default experience, and it's
meant to be. You shouldn't need to hand over a credit card to look at someone's
portfolio project.

If you do want the real model:

```bash
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```

Tests:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Why I built it

I wanted something that would drag me across the whole surface of an AI product
instead of just the interesting bit in the middle. Generating data that behaves
like real data. Writing analysis that survives someone poking at it. Deciding
what the model is and isn't allowed to touch. Making an interface an executive
would actually read. Building the guardrails that make a recommendation safe to
act on.

Healthcare marketing turned out to be a good place to do that, mostly because
the constraints are so obvious. A booking funnel is commercially real, the data
is messy in interesting ways, and the moment anything gets patient-adjacent the
privacy question stops being hypothetical. I got to design around constraints
rather than bolt them on at the end, which is the part I actually wanted
practice at.

This is mine. Not an employer project, not a client project, not modelled on any
particular company's product.

## The bet the whole thing rests on

**Nobody acts on an AI recommendation they can't check.**

So I split the system down the middle: deterministic code calculates every
number, and the model is only ever allowed to interpret them. The copilot gets a
pack of pre-computed aggregates and a system prompt that forbids inventing
figures. Then, and this is the part I'm happiest with, every number in its
answer gets matched back against that pack, and anything that doesn't trace
cleanly gets flagged in the UI as unverified.

I tested this by wiring up a fake model that confidently reports "987,654
bookings lost". The app catches it and says so.

The cost is fluency. The copilot is more clipped than a chatbot that's free to
improvise. What you get back is that every claim is auditable, the thing runs
with no API key, and anyone who doesn't trust the AI can rebuild its numbers from
the dashboard.

## Who it's for

Hospital marketing managers, growth teams, marketing analysts, patient
acquisition teams. More specifically: someone who owns a conversion number, has
to defend a budget, and has maybe two engineering sprints a quarter to spend on
any of it.

Which means the product's real job isn't finding opportunities. It's helping
someone decide what *not* to build.

## What's in the MVP

Four modules:

**Growth Dashboard.** KPI cards, the booking funnel, a drop-off waterfall,
segment comparisons, channel economics, daily trends, and ranked opportunity
sizing. Filterable by date, channel, device and service line.

**AI Conversion Copilot.** Ask a question in plain language. You get an answer
plus a mode badge, an evidence-validation badge, the exact payload that was sent,
and the full prompt if you want it.

**AI Experiment Studio.** Takes a measured gap and turns it into a hypothesis, a
powered A/B design, guardrail metrics, a financial projection with sliders on
every assumption, a seeded simulator, and an approval workflow with an audit
trail.

**AI Trust & Privacy.** The privacy principles, the outbound allowlist, the live
payload, data-quality reporting, a screener you can throw test questions at, and
an evaluation suite you can run yourself.

### Four things I left out on purpose

**Real data connectors.** A live connector would have made the demo flashier and
the privacy story much worse. Synthetic data meant I could build the egress
controls honestly instead of hand-waving them.

**Agentic actions.** The copilot proposes, a named human approves, nothing
launches itself. For anything touching patients I think that's just correct, not
a limitation I'm apologising for.

**Causal claims.** The product compares segments against comparable peers. It
never says it has found a cause, because this data can't find one.

**Invented numbers.** This one rule drove basically the entire architecture,
including a fallback analyst good enough to ship as the default.

## How it's built

```
app.py              Streamlit UI (no business logic lives here)
├── data_generator.py   seeded synthetic dataset with signals planted in it
├── analytics.py        every metric, rate, significance test and projection
├── privacy.py          outbound allowlist, PII screening, data quality
├── ai_analyst.py       prompts, the model call, fallback, number verification
├── experiment.py       design, power analysis, projection, simulation, approvals
└── evaluation.py       analytics invariants and AI behaviour grading
tests/                  168 tests across all six logic modules
```

Dependencies run one way. `app.py` imports everything; nothing imports `app.py`.
The analytics layer has never heard of the AI layer, which is what makes the AI
layer swappable.

**The data.** 90 days across 6 channels, 2 devices and 5 service lines: 5,400
rows, generated from a fixed seed so everyone sees identical numbers. Each funnel
stage is drawn with a binomial from the stage above it, so the funnel can't
contradict itself no matter how I tune the parameters.

I planted a few things in there for the analytics to find: mobile completes
bookings at 44.0% against desktop's 73.0%, one paid channel burns $62.16 per
attended appointment while another manages $42.49, there's a campaign ramp
halfway through the window, weekday seasonality, and roughly 1% of paid-channel
rows have their spend missing. That last one isn't laziness. I wanted the
missing-data handling to have something real to handle.

**Opportunity sizing.** For a given funnel step, the best-converting segment with
enough volume becomes the benchmark. Every other segment gets measured against
it: how many conversions at parity, flowed through that segment's own observed
downstream rates, valued at its own contribution margin. A capture-rate slider
scales the gap, because "reach full parity with desktop" is a ceiling, not a
plan.

**Determinism.** The dataset carries a signature in the sidebar, a hash of its
funnel counts. The analytics layer has no randomness in it at all. The one
stochastic component, the experiment simulator, is seeded and shows you its seed.

## The AI part, and where it falls down

The model gets roughly ten kilobytes of aggregates and a system prompt telling it
to separate observation from hypothesis, state its uncertainty, and recommend one
specific action with a guardrail. Temperature sits at 0.2.

It runs in one of four modes, and the UI always tells you which:

| Mode | When | What happens |
| --- | --- | --- |
| `llm` | Key configured, call worked | The model's answer, grounded in the evidence |
| `fallback` | No key, or offline mode on | Rule-based answer from the same evidence, nothing transmitted |
| `error-fallback` | Key configured, call failed | Rule-based answer, with the error shown rather than buried |
| `blocked` | Question failed screening | Nothing sent, and you're told why |

Now the honest part.

The verifier checks numbers, not reasoning. It'll catch a fabricated figure. It
will not catch a perfectly accurate number deployed in service of a bad
conclusion, and that's the failure mode I'd actually worry about in production.

The evidence pack is a fixed schema, so the model can't go ask the data a
follow-up question. Anything outside what the pack covers gets a partial answer.

The fallback analyst is rule-based, which makes it reliable and reproducible and
also somewhat dim. It recognises the intents I wrote handlers for. Everything
else gets a general overview.

The dataset has no page-level events, no form-field analytics, no session
recordings, no patient outcomes. Several hypotheses the copilot raises are
untestable with the data in scope. It says so, which I'd argue is the right
behaviour, but it's still a real limit.

And I've evaluated how the copilot *behaves*, not whether its advice is any
*good*. Those are different problems and I've only solved the easy one.

## Privacy

Short version: synthetic data only, aggregates out, records never.

The outbound payload is **rebuilt** from an explicit field allowlist rather than
filtered against a blocklist. The difference matters: if someone adds a field
upstream, it's excluded by default until a human puts it on the list. Anything
dropped gets reported in the UI rather than silently disappearing.

Questions are screened before anything is sent, for both personal identifiers and
phrases asking for individual-level data. A match blocks the request outright
instead of quietly redacting it, so you never end up thinking a truncated
question got a full answer.

You can see the exact JSON payload before any request goes out and after every
answer, and download it. It gets re-scanned for PII patterns one last time
immediately before transmission, as a backstop in case I got something wrong
upstream.

Experiments sit in Draft until a named reviewer approves them. Every transition
is recorded.

Keys come from `st.secrets` or `OPENAI_API_KEY`, get masked anywhere they're
displayed, and there's a test asserting no key literal appears in the source.

**All of which is privacy-aware design, not compliance.** I'm not claiming
compliance anywhere in this project and you shouldn't read any into it.

## How I'd know if it worked

I haven't put this in front of users, so these are measures I'd set up, not
results I have:

- **Decision quality.** What share of approved experiments actually ship, and of
  those, how many beat their guardrails.
- **Projection calibration.** Projected lift against measured lift. This is the
  one I'd care about most, because it's the number that would tell me whether the
  sizing model is honest or just confident.
- **Trust behaviour.** How often people open the evidence panel, and whether
  their reliance on the copilot shifts once they have.
- **Time to decision.** Question asked to experiment approved.
- **Grounding rate.** Share of answers clearing verification with nothing
  flagged.

## Where I'd take it next

| Theme | Next step |
| --- | --- |
| Data | Real analytics connectors, with the privacy review happening *before* anything gets built |
| Data | Page-level and form-field events, the biggest hole in the current funnel by a distance |
| Analysis | Causal methods (diff-in-diff, matched segments) instead of parity gaps |
| Analysis | Multi-touch attribution, rather than the last-touch model baked into this dataset |
| Experimentation | Result ingestion, so projections can finally be scored against reality |
| Experimentation | Sequential testing with proper alpha spending, so early stopping stops being dangerous |
| Platform | Persistent storage and real roles for the approval workflow |
| AI | A harness comparing providers on the same evidence pack |

## Built vs. planned

I've tried to keep this line sharp, because portfolio projects have a habit of
blurring it.

**Running in the app right now:** the seeded 5,400-row dataset; funnel, segment,
channel and trend analytics; parity-gap opportunity sizing with an adjustable
capture rate; two-proportion significance testing; the evidence-grounded copilot
with its offline fallback; post-hoc evidence validation; the allowlist, screener
and payload preview; the experiment designer with powered sample sizes and
guardrails; the projection engine; the seeded simulator; the approval workflow
and audit trail; and an evaluation suite you can run from inside the product.

**Not built:** everything in the roadmap table above. None of it exists yet and
nothing in this repo pretends otherwise.

## Tests

```bash
pytest                      # all 168
pytest tests/test_privacy.py -v
python evaluation.py        # the AI evaluation suite, from the command line
```

The suite covers dataset invariants and determinism, analytics correctness and
edge cases, the allowlist and screener, copilot grounding and fallback behaviour
(including a mocked model, a hallucinated figure and a failed call), the
experiment power/projection/simulation/approval logic, and the evaluation harness
itself.

The evaluation suite also runs from inside the app, on the Trust & Privacy page.
That was deliberate. It seemed strange to ask people to trust claims about AI
behaviour that they couldn't verify in the product making them.

## Deploying it

1. Push to GitHub. `.streamlit/secrets.toml` is git-ignored, so a key can't be
   committed by accident.
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   `app.py`.
3. Optionally add secrets in the app settings:
   ```toml
   OPENAI_API_KEY = "sk-..."
   OPENAI_MODEL = "gpt-4o-mini"
   ```
4. Deploy. With no secrets it still runs end to end in fallback mode, which is
   the intended default for anyone just having a look.

The dataset is generated at runtime from a seed, so there's no data file to
upload and no storage to provision.

## Layout

```
Carepath-ai/
├── app.py                      Streamlit UI, 5 pages
├── data_generator.py           synthetic dataset
├── analytics.py                deterministic metrics and projections
├── ai_analyst.py               copilot, fallback, verification
├── experiment.py               design, power, projection, approvals
├── privacy.py                  allowlist, screening, data quality
├── evaluation.py               evaluation harness
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── .streamlit/
│   ├── config.toml             theme
│   └── secrets.toml.example    template (real file is git-ignored)
└── tests/                      168 tests
```

---

Independent personal portfolio project by **Atharv Dabadgaonkar**. Every number
in it is fictional and generated by `data_generator.py`. No real organisation,
patient or visitor is represented anywhere in this repository.
