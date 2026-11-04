<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="V2.0/assets/labviz-logo-dark.svg">
  <img src="V2.0/assets/labviz-logo.svg" alt="LabViz logo" width="320">
</picture>

# LabViz

[English](README.md) · [简体中文](README.zh-CN.md) · [Stable V2.2.0](https://github.com/DDDYT24/lab-data-visualization-tool/releases/tag/v2.2.0) · [V2.3 RC notes](V2.0/RELEASE_NOTES_V2.3.md) · [MIT license](LICENSE)

</div>

LabViz, a scientific plotting tool that runs on your own computer. Hand me a CSV or Excel file and I check it first: missing values, duplicate rows and outliers are flagged, and you decide what stays. Then I turn it into figures you can put in a paper.

## Where I am

- **V2.2.0**: stable. Use this one for everyday work.
- **V2.3.0-rc.2**: preview. The installer is unsigned and acceptance is incomplete, so try it with synthetic data or a copy.
- **Desktop edition** (`desktop/`): I'm learning to run in my own window instead of a browser. Not released yet.

## What I do

- Read CSV, TSV, TXT, JSON and XLSX, and show you a preview before importing.
- Check for missing values, duplicates, type conflicts and outliers. I flag them; I don't delete for you, and every exclusion keeps its reason.
- Draw line, scatter, bar, histogram, box, heatmap and 3D surface charts. V2.3 adds grouped raw points, paired change and violin plots, and up to four panels in one figure.
- Export PNG, SVG and PDF. V2.3 can also freeze a figure with each panel's source data, its parameters and revision history as one research output.
- Keep samples, observations and independent experimental units apart, so repeated measurements never inflate your n.

Everything runs locally. No account, database server or online service is needed.

## Install (Windows)

Download the installer and `SHA256SUMS.txt` from [Releases](https://github.com/DDDYT24/lab-data-visualization-tool/releases) and check the hash first:

```powershell
Get-FileHash .\LabViz-V23-2.3.0-rc.2-Setup.exe -Algorithm SHA256
```

I bring my own Python, Node.js, scientific libraries and fonts. Start **LabViz V2.3** from the Start menu, and use **Stop LabViz V2.3** before a backup. V2.3 and V2.2 keep separate programs and data; you import old data explicitly, and I keep your data when uninstalled.

Installing rc.2 over rc.1 moves your data to a format rc.1 cannot open. If you roll back, I restore the snapshot taken before the upgrade and keep anything saved in rc.2 in a separate retained folder.

See [install, import, backup and uninstall](V2.0/docs/INSTALL_V2.3.md).

## Run from source

You need Git, Python 3.12 or 3.13, and Node.js 22.22.2 or later (packaging uses 24.22.0).

```powershell
git clone https://github.com/DDDYT24/lab-data-visualization-tool.git
Set-Location -LiteralPath .\lab-data-visualization-tool
git checkout v2.3.0-rc.2
.\start-labviz.cmd
```

Source mode uses ports 3001 and 8001 and keeps data in `V2.0/api/.labviz/v23-development`. To rebuild the installer, see the [Windows RC build guide](V2.0/packaging/windows-v23/README.md).

## What I can't promise yet

- Acceptance on a clean machine, a fully disconnected workflow, real 200% zoom with a screen reader, and a physical iPhone.
- Code signing. Windows may warn you; keep its protection on rather than working around it.
- My production dependencies have no known advisories. The lint tools I use in development still have 16 moderate ones through `minimatch`, waiting for an upstream fix; see the [dependency review](V2.0/docs/DEPENDENCY_SECURITY_REVIEW.md).

Whether a figure of mine is ready for a journal depends on you checking its size, fonts, statistical assumptions and caption. I don't judge whether the experiment itself is sound.

## Feedback and license

Report problems in [Issues](https://github.com/DDDYT24/lab-data-visualization-tool/issues), with synthetic data and no personal details in attachments. My source is MIT-licensed; bundled third-party components keep their own licenses, listed in [third-party notices](THIRD_PARTY_NOTICES.md).
