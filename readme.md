# A System Suitability Testing Platform for Untargeted, High-Resolution Mass Spectrometry


## Abstract

Reproducibility of quantitative experiments has been a well-known issue for many omics fields. Within the mass spectrometry community, metabolomics suffers from it the most with lack annotation confidence, standardized normalization techniques and system suitability tests (SST). For untargeted metabolomics, especially utilizing flow injection time-of-flight mass spectrometry (FIA-TOF-MS), batch-to-batch variations are driven extensively by matrix effects and instrument performance. Those are reflected in substantial noise, signal drifts and artefacts that make scientific discoveries confounding.
In this paper, we propose a platform for systematic quality control (QC) and monitoring of instrument performance and properties for FIA-TOF-MS, that consists of a uniquely designed QC sample, an experimental method to capture key aspects of the measurement, and software to engineer and retrieve QC features from raw profile data, analyze them and report on the current machine state through a web-service.
We implemented and deployed a SST platform for the Agilent 6550 iFunnel Q-TOF instrument and collected QC data for 22 months. We found that the QC features, extracted from raw profile data, carry heterogeneous information covering multiple aspects of the measurement and, thus, are relevant in monitoring instrument performance. We present cases of potential use of our SST platform in detecting low resolution or signal-to-noise values. Besides, we observed certain dependencies between QC features and the machine settings, that look promising for at least two future applications: recommending ways to retune an instrument to achieve desired values of QC indicators, and developing a diagnostics system to prevent sudden instrument drops. Finally, we propose an idea to use a QC sample of SST as a reference material in multi-batch longitudinal studies. Serving as a proxy between the measurement and the machine, it can be extremely useful to inform data normalization methods that explicitly take into the account current instrument state.

![Monitoring system snapshot](https://github.com/dmitrav/monitoring_system/blob/master/img/snapshot.png)

The code of the Shiny app can be found [here](https://github.com/dmitrav/shiny_qc).
