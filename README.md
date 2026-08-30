# ad-optimizer

Ad creative optimization. Thompson sampling to decide where the traffic goes,
Claude to write the copy, and a deterministic validator that doesn't believe
either of them.

*This is generalized from client work — the campaign data, account structures
and brand guidelines are gone, and the verticals are made up. The allocation
maths, the platform rules and the validate-then-repair pattern are the real
ones.*

## Two ideas, and they're the same idea

**Don't ask a language model to do arithmetic you can do yourself.**

Claude writes genuinely good ad copy. Claude cannot reliably count to thirty.
Google rejects a 31-character headline, so `specs.py` counts the characters,
and anything over goes back to the model with the exact failure attached —
`headline 1: 40 characters, limit is 30`, not "some headlines are too long".
Two repair rounds, then it returns what passed with the failures attached
rather than looping forever.

That character count is less obvious than it looks. Google and Microsoft
count a full-width CJK character as **two**, so fifteen Japanese characters
is exactly at the thirty-character limit — `len(text)` says fifteen and
happily ships copy that gets rejected on upload. Meta and LinkedIn count code
points. Same string, different length, depending on where you're sending it.

**Don't ask a model something a regex already knows.**

Policy screening runs in two stages. `screen_rules` catches the terms that are
always a problem — guarantees, cure claims, "no credit check" — with no model
call, no latency and no cost. The model only gets consulted for the judgement
calls: implied claims, comparative claims with nothing to compare to, urgency
that misrepresents availability. If stage one already found something that
will definitely be rejected, stage two is skipped, because the copy is going
back to be rewritten either way.

The rule list is deliberately short. A screener with three hundred patterns
flags everything, and a screener that flags everything gets switched off by
the people who have to use it.

## Where the traffic goes

An A/B test splits evenly until it reaches significance, which means it spends
on the loser right up to the last day. `bandit.py` runs Thompson sampling
instead: model each variant's conversion rate as a Beta posterior, draw one
sample from each, serve whichever drew highest. Variants that *might* be good
get explored in proportion to the probability they actually are.

Two guardrails on top, because pure Thompson sampling has a failure mode:

- **`min_impressions`** — nothing gets paused before it's had a fair run,
  whatever the posterior says.
- **`explore_floor`** — every live variant keeps 5%. Without it the sampler
  starves a variant on the strength of a dozen impressions, and a starved
  variant never recovers because it never gets the data that would redeem it.

Everything takes its randomness as a parameter, so a disputed allocation can
be replayed exactly. There's a test that runs 4,000 simulated impressions
against two creatives with known rates and asserts the sampler finds the
better one.

## Testing the LLM parts

The interesting behaviour of an LLM-backed function isn't the LLM. It's
everything around it — whether the validator is trusted over the model,
whether repairs are targeted, whether the loop terminates, whether the brief
got redacted before it went out.

All of that is testable with a stub client. `tests/test_copy_repair.py` covers
the full repair loop, the round limit, refusal handling, and the redaction
pass, and runs offline in under half a second with no API key. 120 tests, none
of which cost anything.

The one thing not covered is whether the copy is any good. That needs human
review and always did.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

To actually generate copy you need `ANTHROPIC_API_KEY` set, or an
`ant auth login` profile — the SDK finds either.

```python
from adopt.copy import generate
from adopt.specs import Platform

result = generate("Managed Postgres for teams who don't want a DBA", Platform.GOOGLE)
print(result.copy.headlines, result.publishable, result.repair_rounds)
```

## Notes on the model layer

`claude-opus-5` with adaptive thinking throughout. Copy generation runs at
`medium` effort — writing headlines isn't a reasoning problem and `high`
produces the same output for more money. The system prompts are cached, which
matters because a campaign generates dozens of these in a batch.

Structured output goes through `client.messages.parse()` with a Pydantic
model, so the response is validated before it reaches the code that uses it.
The policy screener uses the beta endpoint with server-side fallbacks on,
since a prompt full of the exact language it's screening for is a prompt a
safety classifier may reasonably decline — and when the whole chain declines,
it returns "needs a human", not "pass". Failing open on a policy screener is
the wrong direction to fail.

## What's missing

There's no holdout. The bandit optimises within the variants it's given and
has no way to tell you the whole campaign is underperforming — you need a
fraction of traffic on an untouched control for that, and it isn't here.

Conversion feedback is assumed to arrive promptly. Real attribution windows
are days long, so the posterior is always behind, and it's most behind exactly
when a new variant is being evaluated. Delayed-feedback bandits are a real
literature and this doesn't use any of it.

The dedupe on the ledger is per-process. Two workers generating copy for the
same account produce two ledgers and neither knows about the other.

`ALWAYS_PROBLEMATIC` is one list for every vertical. Financial services and
supplements need different lists, and the right shape is a per-vertical
ruleset rather than a global one with exceptions.
