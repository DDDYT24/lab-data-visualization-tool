# Credential and application-key rotation

1. Identify the single secret and consumers. Never rotate database, share-token, and client-identity
   values together.
2. Create the new value through an approved secrets channel; never Terraform, CLI arguments, shell
   history, state, plans, logs, or tickets.
3. For share tokens, add a new versioned key, make it active, retain the previous verification key
   through the maximum token lifetime, then remove it in a later window.
4. For client identity, deploy the new secret to all API tasks together; expect rate-limit identity
   continuity to reset and monitor abuse alarms.
5. For PostgreSQL, rotate the RDS-managed credential, assemble the SQLAlchemy URL in the approved
   secret channel, update Secrets Manager, then replace migration/API/worker tasks in a controlled
   order. Verify connections before retiring the old value.
6. Record only secret ARN/version ID, task revisions, UTC times, tests, and operator—not values.
7. If health fails, restore the previous secret version and task revision; investigate before retry.
