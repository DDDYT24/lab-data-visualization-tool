LabViz turns experimental tables into figures through a guided, local-first workflow. It is designed for people who want to inspect their data and prepare a publication-sized figure without writing plotting code.

## Product strengths

- Import CSV, TSV/TXT, JSON, and Excel workbooks, then review sheets, columns, inferred types, and quality findings before plotting.
- Make cleaning decisions visible and reversible in the project workflow instead of silently changing suspicious values.
- Build common 2D charts and valid gridded X/Y/Z surfaces, with fitting and uncertainty options where the selected method supports them.
- Export figures as PNG, SVG, or PDF and keep exported figure snapshots with the local project history.
- Use the product without an online account: real imported projects and saved figure snapshots are stored in the current computer profile.

## Who LabViz is for

LabViz is intended for students, laboratory researchers, educators, and research teams preparing figures from tabular experimental measurements. It is especially useful when a dataset needs inspection, quality review, a chart, and a reproducible export in one place.

## Example: a 2D time series

This figure was exported by LabViz from the bundled synthetic time-series example. It contains no user or laboratory data.

![Synthetic time-series chart exported by LabViz. Time in minutes is plotted against measured response in millivolts.](/about/response-2d.svg)

[Open the bundled examples](/?examples=1) and choose “Time-series response” to inspect the source table and reproduce the workflow.

## Example: a 3D surface

This figure was exported by LabViz from the bundled synthetic regular-grid X/Y/Z example. A surface is appropriate here because every X/Y grid coordinate is present.

![Synthetic gridded surface exported by LabViz. X and Y are spatial coordinates in millimetres and Z is the measured response in millivolts.](/about/surface-3d.svg)

[Open the bundled examples](/?examples=1) and choose “Regular X/Y/Z surface” to inspect the grid and surface diagnostics.

## Local storage and privacy

LabViz runs its web interface and processing API on this computer. A normal local workflow does not require sign-in, an email service, cloud storage, or an internet connection. The original uploaded file is not retained as an original project attachment; processed project data and metadata are stored in the local LabViz profile. Figure snapshots remain on this computer until you delete them or delete their project. Back up the local LabViz data directory before reinstalling or moving to another computer.

## Current limits

LabViz V2.2.0 is a Windows local release for one ordinary Windows profile. Two-account isolation is untested. Cloud collaboration remains deferred. It does not provide cross-device sync, public sharing, team accounts, or Google/Microsoft login. Quality findings are prompts for review, not automatic scientific conclusions. Statistical tools have method-specific assumptions, and a 3D surface requires a suitable complete grid; inspect diagnostics and the exported result before publication.

## Feedback

Suggestions and bug reports are welcome. This link opens your own email application; LabViz does not send data through a paid email API. Please do not attach confidential experiment data without permission.

[Email liyutao982@gmail.com](mailto:liyutao982@gmail.com)
