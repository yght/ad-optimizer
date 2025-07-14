# 1. Thompson sampling instead of an A/B test

Date: 2025-07-14
Status: Accepted

## Context

The previous process was an even split until significance, then pause the
loser. It works, it is easy to explain to a client, and it spends half the
budget on the losing creative for the entire duration of the test.

On a campaign spending five figures a month, a two week test at an even split
across four creatives puts three quarters of two weeks of budget behind
creatives that will be switched off.

## Decision

Thompson sampling. Each variant's conversion rate is a Beta posterior; each
impression goes to whichever variant draws highest from its own posterior.
Traffic shifts towards what is winning while the test is still running, and
the share a variant earns is exactly the probability that it is the best one.

Two guardrails, because the pure algorithm has a failure mode that matters
here:

- A minimum impression count before anything can be paused. Without it, a
  variant that gets unlucky in its first fifty impressions is written off.
- A 5% explore floor on every live variant. A starved variant never recovers
  on its own, because it never gets the data that would redeem it.

## Consequences

Good:

- Materially less spend on losing creatives. On the first campaign we ran this
  way the losing variants took about a fifth of the traffic they would have.
- "Probability this is the best variant" is a better thing to put in front of
  a client than a p-value, and they ask better questions about it.
- The whole thing is a pure function of the counts, so a disputed allocation
  is replayable from the numbers.

Bad:

- It is harder to explain. "Even split, then pick the winner" fits in a
  sentence; this does not, and some clients want the simpler thing.
- The unequal split makes the per-variant results non-comparable in the naive
  way people want to compare them - a variant with 200 impressions and one
  with 8,000 do not have comparable confidence, and someone will read the
  observed rates side by side anyway.
- No holdout. This optimises within the variants it is given and cannot tell
  you that the whole set is bad.
