# Observability module

Creates the Phase 6C operational dashboard, encrypted alarm notification topic, service alarms,
application/worker log metric filters, and an EventBridge path for S3/KMS access-denied events.

Production objectives are monthly public availability `>=99.9%`, RPO `<=15 minutes`, and RTO
`<=4 hours`. Alarm thresholds are incident signals rather than proof that these objectives have
been measured. The notification endpoint must confirm both the operations and SES-event SNS
subscriptions; a delivered test alarm is required before Phase 6C acceptance.
