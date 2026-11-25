# AI Engine Integration Scaffold

No trained model is shipped in this repository.

## Recommended pipeline

1. Define a bounded initial vocabulary with Deaf/HoH users and an ISL expert.
2. Verify dataset license, consent, collection protocol and permitted usage.
3. Extract hand/body/face features as appropriate for the chosen ISL scope.
4. Train a baseline model with signer-independent train/validation/test splits.
5. Calibrate probabilities and evaluate per-class precision/recall and confusion.
6. Reject low-confidence and out-of-distribution inputs as `unknown`.
7. Test across signers, lighting, camera positions and real-world backgrounds.

## Prediction contract (future)

```json
{
  "label": "HELLO",
  "confidence": 0.0,
  "accepted": false,
  "reason": "model_not_configured"
}
```

The numeric value above is an example schema only, not a real prediction.

## Important

- MediaPipe hand landmarks are features, not sign-language meaning.
- Do not substitute ASL labels or datasets for ISL.
- Do not infer full sentences from isolated-sign classification.
- Include non-manual markers and temporal context when the target signs require them.
- Keep raw video local unless the user explicitly consents to a documented upload.
