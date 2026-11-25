# SignVoice Product Scope

## Initial target

A web-based, accessible communication aid supporting an explicitly documented subset of Indian Sign Language. Start with a small, expert-reviewed vocabulary and fingerspelling only if the dataset and evaluation support it.

## Users and research

Co-design with Deaf and hard-of-hearing people, ISL interpreters and educators. Do not assume one sign is universal across regions or communities. Document regional variants.

## Non-goals for the first release

- Claiming unrestricted sentence-level translation
- Replacing qualified interpreters in legal, medical or emergency settings
- Sending emergency calls or messages without a separately designed, tested integration
- Claiming offline recognition before an offline model is bundled and tested

## Acceptance criteria before showing a recognized sign

- The model is trained on appropriately licensed/consented data.
- Test set includes signers not present in training.
- Per-class metrics and failure cases are documented.
- Confidence is calibrated on held-out data.
- Low-confidence and unknown inputs are rejected.
- Users can correct outputs and report errors.
