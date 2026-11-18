DRAFT — requires human review

# CrowdGPT v2 — Decisions Needed from the Owner

Each item names a recommendation. Nothing here has been acted on.

| # | Decision | Recommendation | Blocks |
|---|---|---|---|
| D1 | Ownership | **Answered 2026-11-17:** community open-source project; the owner contributes as an individual, like an employee working on the Linux kernel. Not Booz Allen work product. A future private deployment on company hardware is a hope, not a plan; keep the design able to run as a private, invite-only network. Owner to confirm any company outside-activity approval through normal channels | — |
| D2 | Outside services and spending | List each one and approve or reject: Hugging Face (mirror), drand (randomness beacon), OpenTimestamps (Bitcoin timestamp anchoring), hosting for the indexer and site, a domain, a CDN, alert channels, cloud GPUs for Phase 0. Recommendation: free services only until there is a budget owner; nothing paid without a named payer | Phases 0, 3 |
| D3 | Token or no token | No token, no wallets, non-transferable credit (`adr/0001`) | Phase 1 record design |
| D4 | Legal and export review | Before publishing weights at any size and before shipping the cryptography code publicly, get a quick review of export rules and of volunteers' locations. The owner arranges this; agents do not decide it | Any public weight release; Phase 4 |
| D5 | Licences | The v1 client and server are GPL-3.0. v1.5 has no `LICENSE` file in this repo. Pick the v2 licence (GPL-3.0 keeps compatibility with v1 code reuse) and record dataset licences per shard. Check the corpus for personal data before E1 | Reusing v1 code; Phase 6 |
| D6 | How strong should identity be | Keys only, with probation and caps. Optional account linking (for example GitHub) for faster probation, as opt-in, because it trades privacy for Sybil resistance | Phase 1.2 |
| D7 | Tokenizer for E1–E4 | Choose a larger vocabulary than v1.5's 4,096 once, before E1 | Phase 6 |
| D8 | Epoch table | Confirm or edit 1B → 4B → 9B → ~27B mixture-of-experts and the token targets in `05` §2 | Phase 5 |
| D9 | Phase 0 hardware | Which GPUs the owner can provide (the M5 Max, the 4060 Ti, others), and whether volunteers are asked to help with cross-GPU noise tests | Phase 0 |
| D10 | Where v2 lives | Options: keep scoping here (local only), a new repo under the owner's account, or propose it upstream to the original CrowdGPT author. Recommendation: talk to the upstream author before any public v2 repo, since v1's community and name are theirs | Any commit or publication |
| D11 | Founding orchestrators | Who runs the first 3–7 orchestrators for Phase 2 (different people, networks, and countries is better) | Phase 2 |
| D12 | Who holds release-signing keys | Proposed 2-of-N among named maintainers | Phase 4 |
| D13 | Contributor hardware floor | Proposed: 16 GB of GPU memory and a minimum upload speed (number set after `TODO.md` 0.11–0.12), fixed in genesis for every epoch (`adr/0005`). Also confirm that epochs past E4 grow as sparse models, not dense, given the compute table in `05` §5.3 | Phase 5 |
