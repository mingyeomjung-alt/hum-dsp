# Security and privacy

Hum DSP is an experimental local file processor, not a network service. Keep
the loopback-only listener local; do not publish it with port forwarding,
tunnels or reverse proxies. No external audio upload or telemetry is added.

Input originals are not intentionally overwritten. Local session input/output
copies can remain after a crash or forced shutdown; cleanup is not secure
erasure. Do not process confidential audio unless you understand and accept
the local storage behavior and have permission to process it.

Do not publish recordings, credentials, session tokens, private filesystem paths
or exploitable security details in public issues. For a security concern,
prefer the repository's private vulnerability-reporting feature **if enabled**.
If it is unavailable, open a minimal issue asking for a private reporting
channel without including the exploit or sensitive data. No response-time or
security-support guarantee is offered.
