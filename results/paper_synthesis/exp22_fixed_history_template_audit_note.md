# Exp22 Fixed-History Template Audit

Run: `exp22_fixed_history_template_dense5_combined_20260506`.

This audit generates one greedy teacher continuation per prompt and replays the same token history through PT raw, IT native-chat, and IT raw/no-template cells. Teacher source is explicit so the IT-native and PT-raw mirrors are never pooled unless requested.

Primary paired raw-lens late-KL results:
Teacher source: `it_native`
- Native fixed effect (`it_native - pt_raw`): `1.354` nats, 95% CI `[1.334, 1.390]`.
- Raw/no-template fixed effect (`it_raw - pt_raw`): `0.656` nats, 95% CI `[0.632, 0.670]`.
- Template delta (`it_native - it_raw`): `0.698` nats, 95% CI `[0.693, 0.730]`.

Quality gates: max malformed rate `0.0000`, missing aligned step rows `0`, minimum CEM retention `0.797`, and maximum post-match SMD `0.079`.
