# Exp54 PT-Teacher Fixed-History Template Audit

Run: `exp54_pt_teacher_dense5_compact_20260506`.

This audit generates one greedy PT-raw teacher continuation per prompt and replays the same token history through PT raw, IT native-chat, and IT raw/no-template cells. Teacher source is explicit so the IT-native and PT-raw mirrors are never pooled unless requested.

Primary paired raw-lens late-KL results:
Teacher source: `pt_raw`
- Native fixed effect (`it_native - pt_raw`): `0.610` nats, 95% CI `[0.135, 1.079]`.
- Raw/no-template fixed effect (`it_raw - pt_raw`): `0.429` nats, 95% CI `[0.155, 0.639]`.
- Template delta (`it_native - it_raw`): `0.181` nats, 95% CI `[-0.105, 0.467]`.

Quality gates: max malformed rate `0.0000`, missing aligned step rows `0`, minimum CEM retention `0.991`, and maximum post-match SMD `0.155`.
