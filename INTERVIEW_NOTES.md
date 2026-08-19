# Interview Notes

1. **Why TF-IDF?** It is fast, interpretable, and effective for a small text baseline.
2. **Why Logistic Regression?** It handles sparse features well and provides class probabilities.
3. **How does TF-IDF work here?** The spam model weights word unigrams/bigrams; the business-intent model weights character n-grams within word boundaries. Both emphasize features common in one message but rarer across the corpus.
4. **Why not classify with an LLM?** The local model is cheaper, faster, reproducible, and demonstrates ML skills.
5. **What does `predict_proba` return?** Each model returns probabilities across its own classes. Spam confidence is the spam probability; a non-spam result returns the selected business-intent probability, conditional on reaching that stage.
6. **What is confidence?** The largest predicted class probability, not a guarantee of correctness.
7. **How is it evaluated?** Exact duplicate SMS messages are removed before sampling. Both models use fixed stratified 80/20 splits. The spam detector has 4,000 training and 1,000 test messages; the business model has 96 training and 24 test emails.
8. **Precision versus recall?** Precision measures correctness of positive predictions; recall measures coverage of true examples.
9. **Why FastAPI?** Validation, typed routing, and automatic OpenAPI documentation.
10. **Why Pydantic?** It validates data and defines clear API contracts.
11. **Why SQLAlchemy?** It keeps persistence readable and database-independent.
12. **Why save the models?** Joblib artifacts let the API start without retraining, and a cached classifier avoids reloading them on every request.
13. **Why rule-based priority?** The rules are explainable and no labelled priority dataset exists.
14. **Why use the LLM only for replies?** Generation benefits from it; classification does not require its cost or variability.
15. **How would production improve?** More representative data, calibration, drift monitoring, privacy controls, and human feedback.
