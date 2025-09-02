# 2. Validate deterministically, then repair

Date: 2025-09-02
Status: Accepted

## Context

The model writes good copy and miscounts characters. Not often, but often
enough that a batch of fifty ads reliably contains a few that are one or two
characters over a limit, and every one of those is a rejection on upload.

Three options were on the table:

1. Ask the model more firmly. Tried it. Stating the limit in the system
   prompt, in the field description, and in the user turn all reduce the rate
   and none of them eliminate it.
2. Truncate the output. Produces headlines that end mid-word.
3. Check it ourselves and send the failures back.

## Decision

Option three. Generated copy is validated by `specs.check_group`, which is
plain code with no model in it, and anything that fails goes back with the
specific failures listed - asset type, index, actual length, allowed length.

Two repair rounds maximum. Then return what there is with the issues attached
rather than continuing to pay for a model that is not converging.

The specificity is the part that matters. "Some headlines are too long"
produces another round of guesses. "headline 1: 40 characters, limit is 30"
gets fixed on the next attempt nearly every time.

## Consequences

Good:

- Copy that reaches the upload step fits, and the rejection rate on length
  went to zero.
- The validator is pure and exhaustively tested, including the CJK
  double-width counting that a naive `len()` gets wrong.
- The repair loop is testable with a stub client. The interesting behaviour -
  targeting, termination, what happens when repair fails - is all covered
  offline.

Bad:

- A repair round is another paid model call. Most generations need none, some
  need one, and the cost is real if the prompt is badly tuned.
- Returning partially-valid copy on failure pushes a decision onto the caller,
  who now has to check `publishable` rather than trusting the return value.
- The model sees its own previous attempt on the repair turn, which means the
  request grows. On a long asset group that is not free.
