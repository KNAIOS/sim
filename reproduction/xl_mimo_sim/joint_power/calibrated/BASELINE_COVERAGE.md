# Coverage check during phase2b training, before development results

The newly calibrated full/coarse selectors use a FOUR-slot action-hold horizon,
because the calibration labels cover one executed decision interval. Their
"full" adjective means all count candidates, not long-horizon optimality.
The original uncalibrated joint search still uses EIGHT slots and is retained.

Consequently, beating calibrated four-slot search alone would NOT establish
superiority over strong calibrated multistep search. A passing bounded screen
must be followed by an eight-slot (or longer) calibrated/robust search baseline
with fair validation and the same current observations, before R1 can be
declared resolved. In particular, gains from anticipating the four-slot wake
delay must not be credited against a baseline denied that horizon.

This coverage note adds a required confirmation check; it does not relax the
already-frozen phase2b screening criteria or change any current training/data.
If the current candidate fails even the one-step/search/PRS comparisons, do
not spend time producing favorable plots from that failure.
