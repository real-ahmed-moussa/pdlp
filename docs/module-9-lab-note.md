# Lab Note: How Retraining Is Triggered in Module 9

The Module 9 lessons start retraining from a Vertex AI Model Monitoring drift alert; the Module 9 labs start the same retraining when a new data span lands in Cloud Storage. Only the first link of the chain differs: everything from the Cloud Run function onward is identical.

## Production design vs this lab

| Step | Production design (lessons) | This lab |
| --- | --- | --- |
| 1. Signal | Vertex AI Model Monitoring detects drift against the training baseline | A new span folder lands in the bucket under `data/span-N/` |
| 2. Event | The alert publishes a message to a Pub/Sub topic | Writing the `_READY` marker file raises a Cloud Storage event |
| 3. Function | Cloud Run function wakes on the Pub/Sub message | The same Cloud Run function wakes on the storage event |
| 4. Guards | Lock against duplicate runs; check the span has train, val and test files | Same |
| 5. Windows | Rebuild training window (latest 3 spans) and evaluation window (latest span) | Same |
| 6. Pipeline | Submit the compiled pipeline to Vertex AI Pipelines | Same |
| 7. Gate and deploy | Evaluator blesses the new model only if it beats the current one; Pusher and the model-push function deploy it | Same |

## Why the lab uses a new span

1. **A real drift alert cannot be produced on demand.** Monitoring compares days of live prediction traffic against the training baseline. A new span can be dropped into the bucket whenever you are ready, and Span 2, set aside since Module 2, plays the role of the drifted data.
2. **An alert alone cannot retrain a model.** The alert says the model is going stale; retraining also needs fresh labeled data. In many production systems the two work together: the alert raises the flag, and the arrival of a new labeled span starts the run.

## Switching to the drift-alert trigger

Two things change, both in the trigger, not the pipeline:

- Deploy the function with a Pub/Sub trigger on the alert topic instead of a Cloud Storage trigger on the bucket.
- Read the span to train on from the alert message (or take the latest complete span) instead of parsing it from the `_READY` file name.

The lock, window building, pipeline submission, blessing gate and deployment code stay as they are in the lab.
