# Exp22 Fixed-History Template Audit

Run: `exp54_holdout_raw_restart_20260506_2000`.

This audit generates one greedy teacher continuation per prompt and replays the same token history through PT raw, IT native-chat, and IT raw/no-template cells. Teacher source is explicit so the IT-native and PT-raw mirrors are never pooled unless requested.

Primary paired raw-lens late-KL results:
Teacher source: `it_native`
- Native fixed effect (`it_native - pt_raw`): `1.181` nats, 95% CI `[1.153, 1.211]`.
- Raw/no-template fixed effect (`it_raw - pt_raw`): `0.549` nats, 95% CI `[0.531, 0.568]`.
- Template delta (`it_native - it_raw`): `0.632` nats, 95% CI `[0.611, 0.654]`.

Teacher source: `pt_raw`
- Native fixed effect (`it_native - pt_raw`): `0.547` nats, 95% CI `[0.506, 0.588]`.
- Raw/no-template fixed effect (`it_raw - pt_raw`): `0.276` nats, 95% CI `[0.241, 0.310]`.
- Template delta (`it_native - it_raw`): `0.272` nats, 95% CI `[0.249, 0.297]`.

Quality gates: max malformed rate `0.0000`, missing aligned step rows `0`, minimum CEM retention `0.999`, and maximum post-match SMD `0.061`.
